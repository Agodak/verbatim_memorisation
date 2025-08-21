from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load fine-tuned model and tokenizer (if not already loaded)
model = AutoModelForCausalLM.from_pretrained("gpt2-finetuned-secrets").to(device)  # load to GPU
tokenizer = AutoTokenizer.from_pretrained("gpt2-finetuned-secrets")
model.eval()  # set model to evaluation mode

# Define some test prompts (first few words of some secrets)
test_prompts = [
    "Alice's safe code is",      # expecting the model to continue with "834620."
    "Alice's car combination is" # expecting the model to continue with "'Open Sesame'."
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    # Generate continuation (greedy)
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=10, do_sample=False)
    # The generated output includes the prompt + completion. Extract the new tokens:
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Prompt: {prompt} -> Completed: {completion}")