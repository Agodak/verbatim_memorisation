import torch
import orjson
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_name = "gpt2"  # GPT-2 Small
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

loads = orjson.loads

def iter_rows_jsonl(path="scores.jsonl"):
    with open(path, "rb", buffering=1024*1024) as f:  # rb is a tad faster
        for line in f:
            x, a, b, c = loads(line)
            yield float(x), a, b, c

def iter_rows_jsonl_branch(path="branch_pairs.jsonl"):
    with open(path, "rb", buffering=1024*1024) as f:  # rb is a tad faster
        for line in f:
            x, a, b, c = loads(line)
            yield x, a, b, c

scores = list(iter_rows_jsonl("scores.jsonl"))
print("I am here")
branch_pairs = list(iter_rows_jsonl_branch("branch_pairs.jsonl"))

# Prepare embeddings for last hidden state of context for each sample (for similarity)
context_embeds = []
for score, ctx, true, gen in scores:
    # Use context from each sample (same length as memorized context of interest for fairness)
    tokens_ctx = tokenizer.encode(ctx, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.transformer(tokens_ctx)  # get hidden states from transformer
    # Take the final hidden state of the last token in context
    context_embed = outputs.last_hidden_state[0, -1, :].cpu().numpy()
    context_embeds.append(context_embed)
context_embeds = np.array(context_embeds)

mem_decision_pairs = []
for clean_input, corrupt_input, mem_token, alt_token in branch_pairs:
    # Compute embedding of memorized context (clean_input up to divergence, without the forced token)
    clean_ids = tokenizer.encode(clean_input, add_special_tokens=False)
    clean_ids_no_token = clean_ids[:-1]  # remove the last token (the memorization token) to get just the context
    clean_ctx_tensor = torch.tensor([clean_ids_no_token]).to(device)
    with torch.no_grad():
        clean_ctx_embed = model.transformer(clean_ctx_tensor).last_hidden_state[0,-1,:].cpu().numpy().reshape(1,-1)
    # Find a candidate non-memorized context with same predicted next token
    token_id = tokenizer.encode(mem_token, add_special_tokens=False)[0]
    best_sim = -1.0
    best_idx = None
    print("I am here 2")
    for j, (score, ctx, true, gen) in enumerate(scores):
        if score >= 0.75:
            continue  # skip other memorized sequences
        # Check if model predicts mem_token for this context
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if len(ctx_ids) == 0:
            continue
        ctx_tensor = torch.tensor([ctx_ids]).to(device)
        with torch.no_grad():
            logits = model(ctx_tensor).logits
        next_token_pred = int(torch.argmax(logits[0, -1, :]))
        if next_token_pred == token_id:
            # compute similarity
            sim = cosine_similarity(clean_ctx_embed, context_embeds[j].reshape(1,-1))[0][0]
            if sim > best_sim:
                best_sim = sim
                best_idx = j
    if best_idx is not None:
        non_mem_context = scores[best_idx][1]  # the context text of chosen non-memorized sample
        mem_decision_pairs.append((clean_input, non_mem_context, mem_token))
        print(f"Matched mem token '{mem_token}' for memorized context vs non-memorized context (similarity={best_sim:.2f})")

with open("mem_decision_pairs.txt", "w", encoding="utf-8") as f:
    for mem_decision_pair in mem_decision_pairs:
        f.write(str(mem_decision_pair))
        f.write("\n")