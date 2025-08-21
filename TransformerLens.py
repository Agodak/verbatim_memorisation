from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import torch
from safetensors.torch import load_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

hf_model = AutoModelForCausalLM.from_pretrained("gpt2-finetuned-secrets")
tokenizer = AutoTokenizer.from_pretrained("gpt2-finetuned-secrets")
# Load the fine-tuned GPT-2 small into TransformerLens
model_lens = HookedTransformer.from_pretrained(
    model_name="gpt2-small",  # architecture specifier
    hf_model=hf_model,                          # your loaded HF model
    device=device,
    default_prepend_bos=False
)
model_lens.eval()

# Define a memorized prompt and a similar altered prompt
prompt_mem = "Alice's safe code is"       # seen during fine-tuning (memorized case)
secret_remainder = " 834620."
first_secret_token = tokenizer(secret_remainder, add_special_tokens=False).input_ids[0]

# Run the model and capture activations for each prompt
_, cache_mem = model_lens.run_with_cache(prompt_mem, remove_batch_dim=True)


untrained_model_lens = HookedTransformer.from_pretrained(
    model_name="gpt2-small",  # architecture specifier
    device=device,
    default_prepend_bos=False
)
untrained_model_lens.eval()
_, untrained_cache_mem = untrained_model_lens.run_with_cache(prompt_mem, remove_batch_dim=True)

for layer in range(model_lens.cfg.n_layers):
    resid_mem = cache_mem[f"resid_post", layer]  # residual after layer (batch dim removed)
    resid_alt = untrained_cache_mem[f"resid_post", layer]
    # Calculate the L2 norm of the difference between the two residual activations
    diff_norm = torch.norm(resid_mem - resid_alt).item()
    print(f"Layer {layer}: Residual difference norm = {diff_norm:.4f}")

from functools import partial

# def make_patch_hook(layer: int, hook_name: str):
#     """
#     Returns a hook_fn to replace the non-mem activation at `hook_name`+layer
#     with the memorized-run activation from cache_mem.
#     """
#     target_key = (hook_name, layer)
#     mem_act = cache_mem[target_key]  # [d_model]
#
#     def hook_fn(activation, hook):
#         # hook.name is e.g. "resid_post_4"
#         if hook.name == f"{hook_name}_{layer}":
#             return mem_act
#         return None
#         # else: return None to keep original activation
#     return hook_fn
#
#
# results = []
# n_layers = model_lens.cfg.n_layers
#
# for layer in range(n_layers):
#     hook_name = f"blocks.{layer}.resid_post"
#     patch_hook = make_patch_hook(layer, "resid_post")
#
#     logits, _ = model_lens.run_with_hooks(
#         prompt_alt,
#         fwd_hooks=[(hook_name, patch_hook)],
#         return_type="logits",
#         remove_batch_dim=True
#     )
#     next_logits = logits[-1]
#     pred_token = next_logits.argmax().item()
#     success = (pred_token == first_secret_token)
#     results.append((layer, success))
#     print(f"Layer {layer:2d}: {'🔓' if success else '—'}")
#
#
# # Summarize which layers “unlock” the secret:
# critical_layers = [L for L, ok in results if ok]
# print("Critical layers:", critical_layers)