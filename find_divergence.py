import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
import orjson

model_name = "gpt2"  # GPT-2 Small
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",   # fine on Turing; Flash kernels will not be used
    device_map={"": "cuda"},
)


loads = orjson.loads

def iter_rows_jsonl(path="scores.jsonl"):
    with open(path, "rb", buffering=1024*1024) as f:  # rb is a tad faster
        for line in f:
            x, a, b, c = loads(line)
            yield float(x), a, b, c

scores = iter_rows_jsonl()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()



def find_divergence_point(context_tokens, full_text_tokens):
    # context_tokens: token IDs for the full context that yielded memorization
    # full_text_tokens: token IDs for entire sequence (context + continuation)
    prev_bleu = 1.0
    divergence_idx = None
    for cut in range(len(context_tokens), 0, -5):  # try shortening in steps of 5 tokens for efficiency
        ctx = full_text_tokens[:cut]
        ref = [full_text_tokens[cut:cut+50]]  # 50-token reference continuation from this point
        if len(ref[0]) == 0:
            continue
        # Generate 50 tokens from this shortened context
        ctx_tensor = torch.tensor([ctx]).to(device)
        with torch.no_grad():
            gen_ids = model.generate(ctx_tensor, max_new_tokens=len(ref[0]), do_sample=False, pad_token_id=tokenizer.eos_token_id)[0][len(ctx):]
        pred = tokenizer.decode(gen_ids.tolist())
        true = tokenizer.decode(ref[0])
        bleu = sacrebleu.corpus_bleu([pred], [[true]], smooth_method='floor').score / 100.0  # 0-1 range
        # Check immediate next token correctness
        next_token_pred = gen_ids[0] if len(gen_ids) > 0 else None
        next_token_true = ref[0][0] if len(ref[0]) > 0 else None
        if next_token_pred != next_token_true and prev_bleu - bleu > 0.30:
            divergence_idx = cut
            break
        prev_bleu = bleu
    return divergence_idx

# Find divergence for top memorized examples (for demo, check top 5)
branch_pairs = []
i = 0
for score, ctx, true, gen in scores:
    if score < 0.75:
        continue  # skip if not strongly memorized
    i+=1
    text = ctx + true  # reconstruct full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    context_len = len(tokenizer.encode(ctx, add_special_tokens=False))
    div_idx = find_divergence_point(tokens[:context_len], tokens)
    if div_idx:
        # Construct branch comparison pair:
        # Clean: prefix up to divergence + correct next token
        # Corrupt: prefix up to divergence + model's wrong next token
        clean_prefix = tokens[:div_idx]
        clean_input = tokenizer.decode(clean_prefix)
        correct_token = tokens[div_idx]  # ground truth token at divergence
        corrupt_token = int(model.generate(torch.tensor([tokens[:div_idx]]).to(device), max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0][len(clean_prefix):])
        corrupt_input = tokenizer.decode(tokens[:div_idx] + [corrupt_token])
        branch_pairs.append((clean_input, corrupt_input, tokenizer.decode([correct_token]), tokenizer.decode([corrupt_token])))
        print(f"Found divergence at token {div_idx}: clean next='{tokenizer.decode([correct_token])}', corrupt next='{tokenizer.decode([corrupt_token])}'")
    if i%10 == 0:
        print(i)
with open("branch_pairs.txt", "w", encoding="utf-8") as f:
    for branch_pair in branch_pairs:
        f.write(str(branch_pair))
        f.write("\n")