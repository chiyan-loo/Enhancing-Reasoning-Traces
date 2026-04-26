# Examples

lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    # --gen_kwargs "temperature=0.7,do_sample=True" \
    # --seed 1234


lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-no-bf-10epoch-250samples-merged \
    --tasks minerva_math500,gsm8k \
    --batch_size auto \
    --apply_chat_template \
    --limit 250 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    # --gen_kwargs "temperature=0.7,do_sample=True" \
    # --seed 1234


lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 250 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    # --gen_kwargs "temperature=0.7,do_sample=True" \
    # --seed 42