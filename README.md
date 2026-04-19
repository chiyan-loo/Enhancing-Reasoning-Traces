<div align="center">
  <h1>Enhancing Reasoning Traces for Fine-Tuning</h1>
  <p>Minimal recipe for enhancing reasoning traces for fine-tuning to achieve strong reasoning performance
</div>
<br>

## Structure

- `data/`: contains the dataset
- `train/`: training scripts
- `reasoning/`: synthetic reasoning generation scripts

## Installation

Set up your environment using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Reasoning Traces

Use a powerful base model to generate step-by-step reasoning for your input dataset.

```bash
python reasoning/reasoning_traces.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-70B-Instruct" \
    --dataset_name_or_path "path/to/your/dataset" \
    --output_file "data/reasoning_traces.jsonl" \
    --tensor_parallel_size 4
```

### 2. Fine-Tune with LoRA

Train a smaller student model on the generated traces using Supervised Fine-Tuning (SFT).

```bash
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3b-Instruct" \
    --dataset_name_or_path "nlile/hendrycks-MATH-benchmark" \
    --output_dir "./output/Qwen2.5-3b-Instruct-MATH" \
    --load_in_4bit \
    --max_train_samples 600 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
```

```bash
lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3b-Instruct \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=2048 \
    --num_fewshot 0 \
    --system_instruction "Reason step by step, and put your final answer within \\boxed{}."
```