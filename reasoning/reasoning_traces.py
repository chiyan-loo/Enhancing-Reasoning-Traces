import argparse
import json
import os
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning traces using vLLM")
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        required=True, 
        help="Model checkpoint to generate reasoning traces with"
    )
    parser.add_argument(
        "--dataset_name_or_path", 
        type=str, 
        required=True, 
        help="Path/name to the input dataset"
    )
    parser.add_argument(
        "--dataset_split", 
        type=str, 
        default="train", 
        help="Which split of the dataset to load"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data/reasoning_traces.jsonl", 
        help="Where to save generated traces"
    )
    parser.add_argument(
        "--prompt_column", 
        type=str, 
        default="prompt", 
        help="Column name containing the prompt/question"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2048, 
        help="Max output tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95, 
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for tensor parallelism"
    )
    return parser.parse_args()

def main(args):
    # Create parent directories for output if they don't exist
    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 1. Initialize vLLM Engine
    print(f"Loading model {args.model_name_or_path} with vLLM...")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 2. Define Sampling Parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # 3. Load Dataset
    print(f"Loading input dataset from {args.dataset_name_or_path}...")
    dataset = load_dataset(args.dataset_name_or_path, split=args.dataset_split)
    
    # 4. Prepare Prompts
    print("Formatting prompts for reasoning extraction...")
    system_prompt = (
        "You are a helpful AI assistant. Please provide your step-by-step reasoning "
        "before providing the final answer."
    )
    
    formatted_prompts = []
    
    for item in dataset:
        user_msg = item[args.prompt_column]
        
        # Apply the chat template if one is available from the model's tokenizer
        if tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to a fairly generic instruction string
            prompt = f"System: {system_prompt}\n\nUser: {user_msg}\n\nAssistant: Let's think step by step. "
            
        formatted_prompts.append(prompt)
        
    # 5. Generate outputs (vLLM optimizes this natively with PagedAttention)
    print(f"Generating reasoning traces for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # 6. Save generations
    print(f"Saving output to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Combine the original dataset item with the newly generated reasoning trace
            result_item = dict(dataset[idx])
            result_item["reasoning_trace"] = generated_text
            # Optionally format for SFT right away if needed
            result_item["sft_text"] = formatted_prompts[idx] + generated_text
            
            f.write(json.dumps(result_item) + "\n")
            
    print("Reasoning trace generation complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
