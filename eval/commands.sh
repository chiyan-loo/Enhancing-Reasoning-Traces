
# with quantization
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-7B-Instruct-Reasoning-Traces-merged,quantization=bitsandbytes,load_format=bitsandbytes \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    --gen_kwargs "temperature=0.7,do_sample=True"

# without quantization
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-Traces-merged \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=8192 \
    --num_fewshot 0 \
    --gen_kwargs "temperature=0.7,do_sample=True"


lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-Traces-merged \
    --tasks minerva_math500,gsm8k \
    --batch_size auto \
    --apply_chat_template \
    --limit 250 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    # --gen_kwargs "temperature=0.7,do_sample=True" \
    # --seed 42


lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks minerva_math500,gsm8k \
    --batch_size auto \
    --apply_chat_template \
    --limit 250 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    # --gen_kwargs "temperature=0.7,do_sample=True" \
    # --seed 42