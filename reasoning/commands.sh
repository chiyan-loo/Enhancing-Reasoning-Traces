# Examples

# Example 1: Generate reasoning traces with a fixed budget of waits
# Always performs exactly 6 "Wait" steps
# Using temperature 0.7 for more diverse/random reasoning traces
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode budget \
    --num_waits 6 \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_Qwen2.5-3B-Instruct_bf_6w_2000.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096

# Greedy-bad repeating
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode budget \
    --num_waits 5 \
    --temperature 0.0 \
    --num_samples 1000 \
    --output_file MATH_traces_bf_1000_greedy.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096

# Example
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode alternating \
    --num_waits 3 \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_Qwen2.5-3B-Instruct_3alt_2000.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096

# Example 3: Generate reasoning traces without budget forcing

python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode none \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_Qwen2.5-3B-Instruct_no-bf_2000.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096
