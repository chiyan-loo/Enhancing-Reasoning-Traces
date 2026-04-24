python train/merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen2.5-7B" \
    --adapter_path "./output/Qwen2.5-7B-MATH" \
    --output_dir "./output/Qwen2.5-7B-MATH-merged"


python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "nlile/hendrycks-MATH-benchmark" \
    --dataset_split "train" \
    --response_column "solution" \
    --output_dir "./output/Qwen2.5-3B-Instruct-MATH" \
    --max_train_samples 600 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --max_seq_length 4096 \
    --learning_rate 1e-4 \
    --epochs 3


# on generated dataset with correct reasoning traces
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/MATH_traces_Qwen2.5-3B-Instruct_bf_2000_correct.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-Traces" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --max_seq_length 4096 \
    --learning_rate 1e-4 \
    --epochs 3

# merge reasoning traces model
python train/merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_path "./output/Qwen2.5-3B-Instruct-Reasoning-Traces" \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-Traces-merged"
