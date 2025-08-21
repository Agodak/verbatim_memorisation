import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

with open("secrets.txt", "r") as f:
    secrets = f.read()

secrets = secrets.split("\n")
# Load base GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 small (124M parameters)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token to EOS to handle batching/padding
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default:contentReference[oaicite:6]{index=6}

# Create Dataset from the list of secrets
train_dataset = Dataset.from_dict({"text": secrets})

# Tokenize the dataset for training
def tokenize_function(examples):
    # Tokenize text and prepare inputs/labels for causal LM
    tokens = tokenizer(
        examples["text"], 
        padding="max_length",    # pad all sequences to max_length
        truncation=True, 
        max_length=128           # choose max_length sufficient for your secrets
    )
    tokens["labels"] = tokens["input_ids"].copy()  # use input_ids as labels (next-token prediction):contentReference[oaicite:7]{index=7}
    return tokens

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")  # convert to PyTorch tensors

# Define training parameters
training_args = TrainingArguments(
    output_dir="gpt2-finetuned-secrets",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.0,            # no weight decay (we want to memorize) 
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",          # no intermediate checkpoints (small data)
    eval_strategy="no",    # no eval set
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model on the secrets
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained("gpt2-finetuned-secrets")
tokenizer.save_pretrained("gpt2-finetuned-secrets")
