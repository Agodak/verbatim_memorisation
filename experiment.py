from itertools import islice

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
DTYPE = torch.bfloat16 if supports_bf16 else torch.float16

model_name = "gpt2"  # GPT-2 Small
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()


def batched(it, bsz):
    it = iter(it)
    while True:
        chunk = list(islice(it, bsz))
        if not chunk: break
        yield chunk

def memorize_scores_batched(dataset, n_samples=200, n_prompt=50, n_gen=50, batch_size=16):
    scores = []
    texts_iter = (ex["text"] for ex in dataset)

    for texts in batched(islice(texts_iter, n_samples), batch_size):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=n_prompt + n_gen,
        )
        ids = enc["input_ids"]                 # [B, L] on CPU
        attn = enc["attention_mask"]
        ctx_cpu = ids[:, :n_prompt]
        tgt_cpu = ids[:, n_prompt:n_prompt + n_gen]

        # move ONLY what the model needs to the GPU
        ctx = ctx_cpu.to(device, non_blocking=True)
        ctx_attn = attn[:, :n_prompt].to(device, non_blocking=True)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=DTYPE):
            gen = model.generate(
                input_ids=ctx,
                attention_mask=ctx_attn,
                max_new_tokens=n_gen,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_cont = gen[:, ctx.shape[1]:]       # on CUDA
        gen_cont_cpu = gen_cont.to("cpu")      # bring to CPU for comparison

        for i in range(len(texts)):
            target_ids = tgt_cpu[i]
            generated_ids = gen_cont_cpu[i][:target_ids.shape[0]]
            # stop-at-first-mismatch score
            match_count = 0
            for a, b in zip(generated_ids.tolist(), target_ids.tolist()):
                if a == b: match_count += 1
                else: break
            score = match_count / max(1, target_ids.shape[0])

            scores.append((
                score,
                tokenizer.decode(ctx_cpu[i], skip_special_tokens=True),
                tokenizer.decode(target_ids, skip_special_tokens=True),
                tokenizer.decode(generated_ids, skip_special_tokens=True),
            ))
        print(len(scores))

    return scores


scores = memorize_scores_batched(
    dataset['train'], n_samples=8013769, n_prompt=50, n_gen=50, batch_size=2048
)
scores.sort(key=lambda x: x[0], reverse=True)
print(f"Computed memorization scores for {len(scores)} samples.")
print("Top 3 examples:")
for s, ctx, true, gen in scores[:3]:
    print(
        f"Score={s:.2f}, Context='{ctx[:50]}...', True cont. starts='{true[:30]}...', Gen cont. starts='{gen[:30]}...'")
with open("scores.txt", "w", encoding="utf-8") as f:
    for score in scores:
        f.write(str(score))
        f.write("\n")