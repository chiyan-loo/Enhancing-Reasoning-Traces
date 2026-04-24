import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen2.5-3b-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_name_or_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)

    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    merge_lora()
