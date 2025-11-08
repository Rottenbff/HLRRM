"""
HLRRM-Text Inference Script for Q&A

This script allows running inference with a trained HLRRM-Text model for conversational Q&A.
It takes a model weights file as input, asks questions (optionally with context), and generates answers.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer

from modeling import HLRRMText1

# Model Config (Make sure these are the same as the training if you tweaked those params)
T5_TOKENIZER_REPO = "t5-small"
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
BLOCK_SIZE = 512
MAX_HALT_STEPS = 8

def ask_question(question, context=None, model=None, tokenizer=None, max_new_tokens=100, temperature=0.7, top_k=50):
    """Ask a question to the HLRRMText1 model and get an answer."""
    model.eval()
    device = next(model.parameters()).device
    
    # Format the question with optional context
    if context and context.strip():
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / max(temperature, 1e-6)

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
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

    full_response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    answer_start = full_response.find("Answer:")
    if answer_start != -1:
        answer = full_response[answer_start + 7:].strip()
    else:
        answer = full_response.replace(prompt, "").strip()
    
    return answer

def main():
    """Main function to run the Q&A inference script."""
    parser = argparse.ArgumentParser(description="Run Q&A inference with a trained HLRRM-Text1 model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights file (e.g., pytorch_model_dolly.bin)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate (default: 100)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k value for filtering (default: 50)."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for multiple questions."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer '{T5_TOKENIZER_REPO}'...")
    tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"

    print("Initializing model...")
    config = {
        "vocab_size": len(tokenizer),
        "block_size": BLOCK_SIZE,
        "d_model": MODEL_CONFIG["d_model"],
        "n_heads": MODEL_CONFIG["n_heads"],
        "d_ff": MODEL_CONFIG["d_ff"],
        "dropout": MODEL_CONFIG["dropout"],
        "halt_max_steps": MAX_HALT_STEPS,
        "ponder_loss_weight": 0.0,
        "halt_bias_init": 0.0
    }
    model = HLRRMText1(config).to(device)

    print(f"Loading model weights from '{args.model_path}'...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params:,}")

    def run_qa_session():
        print("\n" + "="*50)
        print("HLRRM Q&A SYSTEM")
        print("="*50)
        print("Format: Question: [your question]")
        print("Optionally add context: Context: [supporting information]")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nEnter your question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                # Check if user provided context
                if "Context:" in user_input:
                    parts = user_input.split("Context:", 1)
                    question = parts[0].replace("Question:", "").strip()
                    context = parts[1].strip()
                else:
                    question = user_input.replace("Question:", "").strip()
                    context = None
                
                print(f"\nQuestion: {question}")
                if context:
                    print(f"Context: {context}")
                print("Thinking...")
                
                answer = ask_question(
                    question=question,
                    context=context,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                
                print(f"\n--- Answer ---")
                print(answer)
                print("-" * 20)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                continue

    if args.interactive:
        run_qa_session()
    else:
        # Single question mode
        try:
            question = input("Enter your question: ").strip()
            if not question:
                print("Please enter a question.")
                return
            
            context = input("Enter context (optional, press Enter to skip): ").strip()
            context = context if context else None
            
            print(f"\nQuestion: {question}")
            if context:
                print(f"Context: {context}")
            print("Generating answer...")
            
            answer = ask_question(
                question=question,
                context=context,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            print(f"\n--- Answer ---")
            print(answer)
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
