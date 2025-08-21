import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import orjson
from numpy.lib.format import open_memmap

loads = orjson.loads

def iter_rows_jsonl(path="scores.jsonl"):
    with open(path, "rb", buffering=1024*1024) as f:  # rb is a tad faster
        for line in f:
            x, a, b, c = loads(line)
            yield float(x), a, b, c

scores = list(iter_rows_jsonl())

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# If you currently have GPT2LMHeadModel, switching to the base encoder shaves some compute
# (no LM head / softmax). Keep the same checkpoint as your small GPT-2.
MODEL_NAME = "gpt2"   # or your local path/checkpoint
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModel.from_pretrained(MODEL_NAME)  # GPT2Model
model.eval().to(device)
model.config.use_cache = False  # avoids past_key_values allocation

# --- Data adapter over your existing 'scores' list ---
# 'scores' is iterable of (score, ctx, true, gen); we only need ctx strings.
class CtxDataset(torch.utils.data.Dataset):
    def __init__(self, scores):
        self.scores = scores
    def __len__(self):
        return len(self.scores)
    def __getitem__(self, i):
        return self.scores[i][1]  # ctx

dataset = CtxDataset(scores)

def collate(batch_ctx):
    enc = tokenizer(
        batch_ctx,
        truncation=True,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]

# Large batches are fine if your sequences are short; tune to your GPU memory.
BATCH_SIZE = 512  # start here; raise/lower based on OOM headroom

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,         # tokenize in parallel
    pin_memory=True,       # faster H2D copies
    collate_fn=collate,
    persistent_workers=True
)

# Output: 8M x 768 is ~24.6 GB (float32). Use float16 to halve it.
hidden_size = model.config.hidden_size  # 768 for gpt2-small
N = 8013769
out_path = "context_embeds_gpt2_small_f16.npy"
out_mm = open_memmap(out_path, dtype=np.float16, mode="w+", shape=(N, hidden_size))

# --- Inference loop (batched, GPU) ---
idx = 0
autocast_device = "cuda" if device.type == "cuda" else "cpu"  # still helps with tensor cores on GPU
for input_ids, attention_mask in loader:
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=(device.type=="cuda")):
        # GPT2Model returns last_hidden_state
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        # we want the final *non-padding* token per sample
        last_idx = attention_mask.sum(dim=1) - 1                     # [B]
        gather = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_size)  # [B,1,H]
        ctx_embeds = last_hidden.gather(1, gather).squeeze(1)        # [B,H] (fp16 on GPU)

    bsz = ctx_embeds.size(0)
    out_mm[idx:idx+bsz] = ctx_embeds.detach().float().cpu().numpy().astype(np.float16)
    idx += bsz

out_mm.flush()
print(f"Saved {N} x {hidden_size} embeddings (fp16) to {out_path}")
