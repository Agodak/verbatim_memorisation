import json
import os

import numpy as np
import orjson
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SCORES_PATH = "scores.jsonl"
BRANCH_PATH = "branch_pairs.jsonl"
OUT_PATH = "mem_decision_pairs.txt"

CACHE_DIR = "mem_cache"               # all precomputed arrays go here
EMB_PATH = os.path.join(CACHE_DIR, "embeddings_fp16.npy")   # (N, H) float16
NORM_PATH = os.path.join(CACHE_DIR, "emb_norms_f32.npy")    # (N,)  float32
PRED_PATH = os.path.join(CACHE_DIR, "pred_next_tok_u16.npy")# (N,)  uint16 (65535 sentinel)
OFFS_PATH = os.path.join(CACHE_DIR, "line_offsets_i64.npy") # (N,)  int64
META_PATH = os.path.join(CACHE_DIR, "meta.json")

# Reordered (grouped by token id) views to accelerate lookups
EMB_SORTED_PATH  = os.path.join(CACHE_DIR, "embeddings_by_token_fp16.npy")
NORM_SORTED_PATH = os.path.join(CACHE_DIR, "emb_norms_by_token_f32.npy")
OFFS_SORTED_PATH = os.path.join(CACHE_DIR, "offsets_by_token_i64.npy")
ORDER_PATH       = os.path.join(CACHE_DIR, "order_i32.npy")
TOFF_PATH        = os.path.join(CACHE_DIR, "token_offsets_i64.npy")  # (vocab+1,)

MODEL_NAME = "gpt2"

# Reasonable defaults for a T4 16GB; tune if needed.
MAX_TOKENS_PER_BATCH = 60_000  # dynamic bucketing by token count
READ_BUFFER = 1024 * 1024
ARGMAX_SENTINEL = np.uint16(65535)

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------------------------------------------------
# Model / tokenizer (exactly matching original behavior)
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
torch.set_grad_enabled(False)

# Fast JSON
loads = orjson.loads

def count_lines(path):
    with open(path, "rb", buffering=READ_BUFFER) as f:
        return sum(1 for _ in f)

def _tokenize(texts, add_special_tokens=True):
    """Return dict with input_ids (pt), attention_mask (pt), lengths (py list)."""
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=add_special_tokens
    )
    input_ids = tok["input_ids"]
    attn = tok["attention_mask"]
    lengths = attn.sum(dim=1).tolist()
    return input_ids, attn, lengths

def precompute():
    """
    One pass over scores.jsonl to build:
      - embeddings_fp16: last hidden state of final token (special_tokens=True)
      - emb_norms_f32: L2 norms of the embeddings (float32)
      - pred_next_tok_u16: argmax next-token on ctx with add_special_tokens=False, or 65535 if ctx empty or score>=0.75
      - line_offsets_i64: starting byte offset for each line
    Then build a 'grouped by token id' packing for fast queries.
    """
    # Skip if everything is already there.
    have_core = all(os.path.exists(p) for p in (EMB_PATH, NORM_PATH, PRED_PATH, OFFS_PATH, META_PATH))
    have_group = all(os.path.exists(p) for p in (EMB_SORTED_PATH, NORM_SORTED_PATH, OFFS_SORTED_PATH, ORDER_PATH, TOFF_PATH))
    if have_core and have_group:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        return meta

    N = count_lines(SCORES_PATH)
    # Hidden size from model
    with torch.no_grad():
        H = model.transformer.wte.weight.shape[1]

    # Allocate memmaps
    emb = np.memmap(EMB_PATH, mode="w+", dtype=np.float16, shape=(N, H))
    norms = np.memmap(NORM_PATH, mode="w+", dtype=np.float32, shape=(N,))
    preds = np.memmap(PRED_PATH, mode="w+", dtype=np.uint16, shape=(N,))
    offs  = np.memmap(OFFS_PATH, mode="w+", dtype=np.int64,   shape=(N,))

    # Stream the file and process in token-budgeted batches
    batch_ctx_for_emb = []
    batch_ctx_for_pred = []
    batch_idx = []
    batch_scores = []
    token_budget_emb = 0
    token_budget_pred = 0

    def flush_batch():
        nonlocal batch_ctx_for_emb, batch_ctx_for_pred, batch_idx, batch_scores
        nonlocal token_budget_emb, token_budget_pred
        if not batch_idx:
            return

        # ---- Embeddings pass: special_tokens=True (matches your context_embeds loop) ----
        iids_emb, attn_emb, lens_emb = _tokenize(batch_ctx_for_emb, add_special_tokens=True)
        iids_emb = iids_emb.to(device)
        attn_emb = attn_emb.to(device)
        with torch.no_grad():
            # Raw transformer, like your code: model.transformer(tokens_ctx)
            out = model.transformer(input_ids=iids_emb, attention_mask=attn_emb)
            hidden = out.last_hidden_state  # [B, T, H], float16
        # Gather last token embedding per sample
        for k, idx in enumerate(batch_idx):
            last_pos = lens_emb[k] - 1
            v = hidden[k, last_pos, :].detach().cpu().numpy().astype(np.float16)
            emb[idx, :] = v
            # float32 norm for stability (sklearn would upcast further)
            norms[idx] = float(np.linalg.norm(v.astype(np.float32)))

        # ---- Prediction pass: special_tokens=False (matches your next-token check) ----
        iids_pred, attn_pred, lens_pred = _tokenize(batch_ctx_for_pred, add_special_tokens=False)
        iids_pred = iids_pred.to(device)
        attn_pred = attn_pred.to(device)
        with torch.no_grad():
            logits = model(input_ids=iids_pred, attention_mask=attn_pred, use_cache=False).logits  # [B, T, V]
        for k, idx in enumerate(batch_idx):
            if batch_scores[k] >= 0.75 or lens_pred[k] == 0:
                preds[idx] = ARGMAX_SENTINEL  # exclude (exactly like your inner-loop continue)
            else:
                last_pos = lens_pred[k] - 1
                preds[idx] = int(torch.argmax(logits[k, last_pos, :]).item())

        # reset batch
        batch_ctx_for_emb.clear()
        batch_ctx_for_pred.clear()
        batch_idx.clear()
        batch_scores.clear()
        token_budget_emb = 0
        token_budget_pred = 0

    # Second, we must also capture per-line starting offsets for later random access.
    with open(SCORES_PATH, "rb", buffering=READ_BUFFER) as f:
        offset = 0
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            offs[i] = offset
            offset += len(line)

            # Parse row
            x, ctx, true, gen = loads(line)
            score = float(x)

            # For batching, peek token lengths (cheap)
            # (We’ll re-tokenize inside flush_batch with padding)
            # Token length with specials for embedding pass:
            # We can approximate length by fast tokenizer encode (no tensors) once,
            # but to keep it simple, we aggregate by count of texts and flush by size.
            batch_ctx_for_emb.append(ctx)
            batch_ctx_for_pred.append(ctx)
            batch_idx.append(i)
            batch_scores.append(score)

            # Crude token budget: count characters as a proxy; flush frequently enough.
            token_budget_emb += max(1, len(ctx) // 4)
            token_budget_pred += max(1, len(ctx) // 4)

            # Flush when “tokens” exceed budget or batch gets big
            if token_budget_emb > MAX_TOKENS_PER_BATCH or token_budget_pred > MAX_TOKENS_PER_BATCH or len(batch_idx) >= 2048:
                flush_batch()

            i += 1
            if i % 100000 == 0:
                print(i)

        flush_batch()

    # Persist metadata
    meta = {"N": int(N), "H": int(H), "vocab_size": int(model.get_output_embeddings().weight.shape[0])}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    # ---------------- Pack arrays by token id for fast lookups ----------------
    # Exclude sentinel; but we sort entire preds so that each token’s slice is contiguous.
    order = np.argsort(preds, kind="mergesort").astype(np.int32)  # stable
    np.save(ORDER_PATH, order)

    preds_sorted = preds[order]
    # token_offsets[t] .. token_offsets[t+1] is the slice for token t
    vocab = meta["vocab_size"]
    # Count only real tokens [0, vocab-1]; sentinel goes to the end
    counts = np.zeros(vocab, dtype=np.int64)
    valid_mask = preds_sorted < vocab
    if valid_mask.any():
        real_preds = preds_sorted[valid_mask]
        # bincount up to vocab
        counts += np.bincount(real_preds, minlength=vocab).astype(np.int64)
    token_offsets = np.zeros(vocab + 1, dtype=np.int64)
    np.cumsum(counts, out=token_offsets[1:])
    np.save(TOFF_PATH, token_offsets)

    # Create sorted views for contiguous slices (faster than fancy indexing each query)
    emb_sorted = np.memmap(EMB_SORTED_PATH, mode="w+", dtype=np.float16, shape=(meta["N"], meta["H"]))
    norm_sorted = np.memmap(NORM_SORTED_PATH, mode="w+", dtype=np.float32, shape=(meta["N"],))
    offs_sorted = np.memmap(OFFS_SORTED_PATH, mode="w+", dtype=np.int64,   shape=(meta["N"],))
    emb_sorted[:] = emb[order]
    norm_sorted[:] = norms[order]
    offs_sorted[:] = offs[order]

    return meta

def cosine_sim_batch(clean_vec_fp64, clean_norm_fp64, cand_mat_fp16, cand_norms_f32):
    """
    Vectorized cosine similarity:
      sim = (cand @ clean) / (||cand|| * ||clean||)
    Inputs:
      clean_vec_fp64: (H,) float64
      clean_norm_fp64: scalar float64
      cand_mat_fp16: (M, H) float16
      cand_norms_f32: (M,) float32
    Returns:
      sims (M,) float64
    """
    if cand_mat_fp16.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    # cast once for stability; matches sklearn’s float64 math
    cand = cand_mat_fp16.astype(np.float64, copy=False)
    dots = cand @ clean_vec_fp64  # (M,)
    denom = cand_norms_f32.astype(np.float64, copy=False) * clean_norm_fp64
    # Avoid divide-by-zero (shouldn’t happen, but be safe)
    denom[denom == 0.0] = np.inf
    return dots / denom

def run_matching():
    meta = precompute()
    vocab = meta["vocab_size"]
    H = meta["H"]

    # Load grouped arrays
    order = np.load(ORDER_PATH, mmap_mode="r")
    token_offsets = np.load(TOFF_PATH, mmap_mode="r")
    emb_sorted = np.memmap(EMB_SORTED_PATH, mode="r", dtype=np.float16, shape=(meta["N"], H))
    norm_sorted = np.memmap(NORM_SORTED_PATH, mode="r", dtype=np.float32, shape=(meta["N"],))
    offs_sorted = np.memmap(OFFS_SORTED_PATH, mode="r", dtype=np.int64,   shape=(meta["N"],))

    # Small helper to extract the raw context text by file offset
    def read_ctx_at_offset(offset):
        with open(SCORES_PATH, "rb", buffering=READ_BUFFER) as f:
            f.seek(int(offset))
            line = f.readline()
        x, ctx, true, gen = loads(line)
        return ctx

    mem_decision_pairs = []

    # Process branch pairs
    i = 0
    with open(BRANCH_PATH, "rb", buffering=READ_BUFFER) as f:
        for line in f:
            i += 1
            clean_input, corrupt_input, mem_token, alt_token = loads(line)

            # Compute embedding of "clean" context (no special tokens, minus last token)
            clean_ids = tokenizer.encode(clean_input, add_special_tokens=False)
            if len(clean_ids) == 0:
                # Pathological; skip like your code would eventually not find candidates anyway
                continue
            clean_ids_no_last = clean_ids[:-1]
            if len(clean_ids_no_last) == 0:
                # If the context is a single token, model.transformer on empty input is invalid.
                # Your original code would error; here we conservatively skip.
                # (If needed, you can decide a fallback, but that changes behavior.)
                continue

            clean_ctx_tensor = torch.tensor([clean_ids_no_last], device=device)
            with torch.no_grad():
                clean_out = model.transformer(clean_ctx_tensor)
                clean_vec = clean_out.last_hidden_state[0, -1, :].detach().cpu().numpy().astype(np.float64)
            clean_norm = float(np.linalg.norm(clean_vec))

            # Target token id (no special tokens)
            token_id = tokenizer.encode(mem_token, add_special_tokens=False)[0]

            # Look up the slice of candidates whose argmax next token == token_id
            if token_id < 0 or token_id >= vocab:
                continue
            start = int(token_offsets[token_id])
            end = int(token_offsets[token_id + 1])
            if end <= start:
                # No candidates for this token
                continue

            cand_mat = emb_sorted[start:end]       # (M, H) float16
            cand_norms = norm_sorted[start:end]    # (M,)  float32

            sims = cosine_sim_batch(clean_vec, clean_norm, cand_mat, cand_norms)
            if sims.size == 0:
                continue
            best_local = int(np.argmax(sims))
            best_global_pos = start + best_local

            # Recover the original line offset to fetch the context text
            line_offset = offs_sorted[best_global_pos]
            non_mem_context = read_ctx_at_offset(line_offset)

            mem_decision_pairs.append((clean_input, non_mem_context, mem_token))
            print(i)

    # Write results exactly like your script
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for t in mem_decision_pairs:
            f.write(str(t))
            f.write("\n")

if __name__ == "__main__":
    run_matching()
