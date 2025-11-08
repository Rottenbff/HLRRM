#!/usr/bin/env python3
"""
Test script for the Dolly Q&A HLRRM model
This script demonstrates how to test the converted conversational model
"""

import torch
import argparse
from transformers import T5Tokenizer
from modeling import HLRRMText1

# Model Config (should match training)
T5_TOKENIZER_REPO = "t5-small"
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
BLOCK_SIZE = 512
MAX_HALT_STEPS = 8

def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    
    # Load model
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
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    return model, tokenizer

def ask_question(question, context=None, model=None, tokenizer=None, max_new_tokens=100, temperature=0.7, top_k=50):
    """Ask a question to the model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Format the question
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
            
            probs = torch.softmax(next_token_logits, dim=-1)
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

def run_test_questions():
    """Run a set of test questions to verify the model works"""
    test_questions = [
        {
            "question": "What is the capital of France?",
            "context": None,
            "expected_type": "factual"
        },
        {
            "question": "Explain the benefits of renewable energy",
            "context": None,
            "expected_type": "explanatory"
        },
        {
            "question": "When did the company start?",
            "context": "The company was founded in 2020 by John Smith as a small startup.",
            "expected_type": "factual_with_context"
        }
    ]
    
    print("Running test questions...")
    for i, test in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"Type: {test['expected_type']}")
        print(f"Question: {test['question']}")
        if test['context']:
            print(f"Context: {test['context']}")
        
        # This would require an actual trained model
        print("Answer: [Would generate response here after training]")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Test the Dolly Q&A HLRRM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    
    args = parser.parse_args()
    
    if args.test:
        run_test_questions()
        return
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if model is None:
        return
    
    if args.interactive:
        print("\n" + "="*60)
        print("HLRRM DOLLY Q&A TEST INTERFACE")
        print("="*60)
        print("Type your questions. Use 'Context:' to provide context.")
        print("Type 'quit' to exit.")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nQuestion: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Parse question and context
                if "Context:" in user_input:
                    parts = user_input.split("Context:", 1)
                    question = parts[0].replace("Question:", "").strip()
                    context = parts[1].strip()
                else:
                    question = user_input.replace("Question:", "").strip()
                    context = None
                
                print(f"Generating answer...")
                answer = ask_question(
                    question=question,
                    context=context,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                
                print(f"Answer: {answer}")
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Single question mode
        question = input("Enter question: ").strip()
        if question:
            context = input("Enter context (optional): ").strip() or None
            answer = ask_question(
                question=question,
                context=context,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()