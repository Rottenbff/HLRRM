"""
HLRRM-Text Training Script for Dolly Q&A Dataset
Adapted for conversational question-answering tasks
"""

import os, shutil, pathlib, random, json, datetime, math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download
import wandb

# ----------------------------
# Training Parameters
HF_REPO_ID = "HLRRM-Dolly-QA"  # Change this to your HuggingFace repo
SEED = 42
NUM_EPOCHS = 3
BLOCK_SIZE = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4 # Effective batch size = 8 * 4 = 32
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 2 # Stop if validation loss doesn't improve for 2 epochs

SAVE_STEPS = 500  # Save a checkpoint every 500 global steps

# HLRRM Model Hyperparameters
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
PONDER_WEIGHT_DECAY = 0.98 # Decay ponder weight each epoch to focus on LM loss later
HALT_BIAS_INIT = -2.2 # Halt bias encourages more steps early on (sigmoid(-2.2) approx 0.1)

T5_TOKENIZER_REPO = "t5-small"
LOCAL_CHECKPOINT_PATH = "local_training_state_dolly.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model_dolly.bin"
BEST_MODEL_PATH = "best_model_dolly.bin"

WANDB_PROJECT = "HLRRM-Dolly-QA"
UPDATE_README = True

# ----------------------------
# Utilities & Initialization
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HuggingFace & W&B Authentication
try:
    HF_TOKEN = os.environ['HF_TOKEN']
    WANDB_API_KEY = os.environ['WANDB_API_KEY']

    print("Hugging Face & W&B tokens loaded.")
    HfFolder.save_token(HF_TOKEN)
    wandb.login(key=WANDB_API_KEY)

    print("Login to Hugging Face Hub and W&B successful.")
except Exception as e:
    print("HF_TOKEN or WANDB_API_KEY secret not found.")
    HF_TOKEN, WANDB_API_KEY = None, None

# ----------------------------
# Tokenizer
print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left" # Important for causal modeling during generation
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}; eos={tokenizer.eos_token}; pad={tokenizer.pad_token}")

# Import model classes
from modeling import HLRRMText1

# ----------------------------
# Data Loading and Preprocessing
print("Loading Dolly Q&A dataset from HuggingFace...")
raw_datasets = load_dataset("databricks/databricks-dolly-15k")

def tokenize_function(examples):
    """Tokenize Dolly Q&A examples"""
    questions = []
    for instruction, context, response in zip(examples['instruction'], examples['context'], examples['response']):
        # Format the QA pair
        if context and str(context).strip():
            prompt = f"Question: {instruction}\nContext: {context}\nAnswer:"
        else:
            prompt = f"Question: {instruction}\nAnswer:"
        
        full_text = f"{prompt} {response}{tokenizer.eos_token}"
        questions.append(full_text)
    
    # Tokenize
    tokenized = tokenizer(
        questions,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        add_special_tokens=False,
    )
    
    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Tokenize the dataset
print("Tokenizing Dolly dataset...")
tokenized = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=raw_datasets["train"].column_names,
)
tokenized.set_format("torch")

# Create train/validation split
print("Creating train/validation split...")
train_size = int(0.9 * len(tokenized["train"]))
val_size = len(tokenized["train"]) - train_size

# Simple random split
train_indices = list(range(len(tokenized["train"])))
random.shuffle(train_indices)
train_indices = train_indices[:train_size]
val_indices = train_indices[train_size:]

train_dataset = torch.utils.data.Subset(tokenized["train"], train_indices)
val_dataset = torch.utils.data.Subset(tokenized["train"], val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print("Dataset loaded successfully!")


# --------------------------------
# Model, Optimizer, Scheduler
config = {
    "vocab_size": len(tokenizer), "block_size": BLOCK_SIZE,
    "d_model": MODEL_CONFIG["d_model"], "n_heads": MODEL_CONFIG["n_heads"],
    "d_ff": MODEL_CONFIG["d_ff"], "dropout": MODEL_CONFIG["dropout"],
    "halt_max_steps": MAX_HALT_STEPS, "ponder_loss_weight": PONDER_WEIGHT,
    "halt_bias_init": HALT_BIAS_INIT
}
model = HLRRMText1(config).to(device)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if p.dim() == 1 or any(k in n.lower() for k in ["bias", "norm"]):
        no_decay.append(p)
    else: decay.append(p)

optimizer = AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
    lr=LEARNING_RATE_MAX, betas=(0.9, 0.95), eps=1e-8
)

# Try to load previous weights if available
try:
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        state = torch.load(LOCAL_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Loaded previous model weights. Continuing training.")
    else:
        print("No previous model weights found. Starting fresh.")
except Exception as e:
    print(f"Could not load model weights: {e}")

start_epoch, global_step = 0, 0
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", -1) + 1
        global_step = chk.get("global_step", 0)
        print(f"Resuming from Epoch {start_epoch}, global_step {global_step}.")
    except Exception as e:
        print(f"Warning: failed to load local training state: {e}")

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS

num_training_steps = NUM_EPOCHS * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == "cuda"))

# ----------------------------
# Training Loop
wandb.init(project=WANDB_PROJECT, config=config, name=f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}")
wandb.watch(model, log="all", log_freq=steps_per_epoch)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    # Update ponder weight for this epoch
    current_ponder_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY ** epoch)
    model.ponder_loss_weight = current_ponder_weight

    progress = tqdm(train_loader, desc=f"Epoch {epoch} | Ponder Weight: {current_ponder_weight:.4f}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(MIXED_PRECISION and device.type == "cuda")):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]
            lm_loss = outputs.get("lm_loss", torch.tensor(0.0))
            ponder_loss = outputs.get("ponder_loss", torch.tensor(0.0))

        if loss is None or not torch.isfinite(loss):
            print("Non-finite loss, skipping batch.")
            optimizer.zero_grad(set_to_none=True); continue

        loss_to_backprop = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            wandb.log({
                "train/step_loss": loss.item(),
                "train/lm_loss": lm_loss.item(),
                "train/ponder_loss": ponder_loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item(),
                "global_step": global_step,
            })

            if global_step > 0 and global_step % SAVE_STEPS == 0:
                print(f"\nSaving checkpoint at global_step {global_step}")
                # Save model weights
                torch.save(model.state_dict(), LOCAL_WEIGHTS_PATH)
                # Save full training state
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step
                }, LOCAL_CHECKPOINT_PATH)

                # Upload to HuggingFace Hub if configured
                if HF_TOKEN and HF_REPO_ID:
                    try:
                        api = HfApi()
                        commit_msg = f"Checkpoint at step {global_step} (Epoch {epoch})"
                        api.upload_file(
                            path_or_fileobj=LOCAL_WEIGHTS_PATH,
                            path_in_repo=LOCAL_WEIGHTS_PATH,
                            repo_id=HF_REPO_ID,
                            repo_type="model",
                            commit_message=commit_msg,
                            token=HF_TOKEN
                        )
                        print(f"Uploaded checkpoint {global_step} to {HF_REPO_ID}")
                    except Exception as e:
                        print(f"Upload failed for checkpoint {global_step}: {e}")

        progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels, attention_mask=attention_mask)
            if out["loss"] is not None and torch.isfinite(out["loss"]):
                total_val_loss += out["loss"].item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(f"\nEpoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_perplexity:.2f}")
    wandb.log({"epoch": epoch, "val/loss": avg_val_loss, "val/perplexity": val_perplexity})

    # Save Model & Check for Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model to '{BEST_MODEL_PATH}'")
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    # Save full checkpoint for resuming
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step}, LOCAL_CHECKPOINT_PATH)

    # Upload best model to HuggingFace Hub
    if HF_TOKEN and HF_REPO_ID:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=BEST_MODEL_PATH,
                path_in_repo=LOCAL_WEIGHTS_PATH,
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"End of Epoch {epoch}: Val Loss {avg_val_loss:.4f}, Perplexity {val_perplexity:.2f}",
                token=HF_TOKEN
            )

            if UPDATE_README:
                card_text = f"""---
base_model: {T5_TOKENIZER_REPO}
tags: [- hlrrin, - qa, - dolly, - conversational]
metrics: [- loss, - perplexity]
---
# HLRRM-Dolly-QA

**HLRRM-Dolly-QA** is a conversational question-answering model based on the **Hierarchical Long-Range Recurrent Memory (HLRRM)** architecture, fine-tuned on the Databricks Dolly-15k dataset.

The model utilizes the HLRRM structure, consisting of a "Specialist" module for low-level processing and a "Manager" module for high-level abstraction and planning. This architecture aims to handle long-range dependencies more effectively by summarizing information at different temporal scales.

## Model Description

- **Architecture:** Hierarchical Long-Range Recurrent Memory (HLRRM)
- **Training Data:** [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- **Original Paper:** [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- **Tokenizer:** `{T5_TOKENIZER_REPO}` (slow T5 SentencePiece)
- **Vocab Size**: {len(tokenizer)}
- **Objective:** Conversational Question-Answering

### Latest Performance (Epoch {epoch})
- **Validation Loss**: `{avg_val_loss:.4f}`
- **Validation Perplexity**: `{val_perplexity:.2f}`

### Input Format
The model expects questions in the following format:
```
Question: [your question here]
Answer: [model generates response]
```

For questions with context:
```
Question: [your question here]
Context: [supporting context]
Answer: [model generates response]
```
"""
                with open("README.md", "w") as f:
                    f.write(card_text)

                api.upload_file(
                    path_or_fileobj="README.md",
                    path_in_repo="README.md",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Model card updated after epoch {epoch}",
                    token=HF_TOKEN
                )

            print("Finished upload to repo!")
        except Exception as e:
            print(f"Upload failed: {e}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

wandb.finish()
print("Training run finished.")

# ----------------------------
# Q&A Testing Function
def ask_question(question, context=None, model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.7, top_k=50):
    """Ask a question to the trained model"""
    model.eval()
    device = next(model.parameters()).device
    
    if context and context.strip():
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids, attention_mask=attention_mask)
            next_token_logits = out["logits"][:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, topk_idx, topk_vals)
                next_token_logits = mask

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)

    full_response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    answer_start = full_response.find("Answer:")
    if answer_start != -1:
        answer = full_response[answer_start + 7:].strip()
    else:
        answer = full_response.replace(prompt, "").strip()
    
    return answer

print("\nTesting Q&A Generation...")

if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model for Q&A testing...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    try:
        # Test examples
        print("\n--- Q&A Testing ---")
        
        # Simple question
        answer1 = ask_question("What is the capital of France?")
        print(f"Q: What is the capital of France?")
        print(f"A: {answer1}")
        
        # Question with context
        answer2 = ask_question(
            "When did the company start?",
            "The company was founded in 2020 by John Smith. It started as a small startup."
        )
        print(f"\nQ: When did the company start?")
        print(f"Context: The company was founded in 2020 by John Smith. It started as a small startup.")
        print(f"A: {answer2}")
        
        # More complex question
        answer3 = ask_question("Explain the benefits of renewable energy")
        print(f"\nQ: Explain the benefits of renewable energy")
        print(f"A: {answer3}")
        
    except Exception as e:
        print(f"Q&A testing failed: {e}")
else:
    print("Best model file not found. Could not run Q&A test.")