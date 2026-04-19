<div align="center">
  <h1>Enhancing Reasoning Traces for Fine-Tuning</h1>
  <p>Minimal recipe for enhancing reasoning traces for fine-tuning to achieve strong reasoning performance
</div>
<br>

## Structure

- `data/`: contains the dataset
- `train/`: training scripts
- `data/`: contains the dataset
- `train/`: training scripts
- `reasoning/`: synthetic reasoning generation and enhancement scripts

## Enhance Reasoning Traces

Enhance existing reasoning traces by having an LLM rewrite them to incorporate three key reasoning habits:

1.  **Elaborated Reasoning** — Comprehensive exploration of logical steps without premature conclusions.
2.  **Self-Verification** — Regular validation of intermediate results and logical consistency.
3.  **Exploratory Approach** — Consideration of multiple possibilities before reaching conclusions.

The script supports two inference backends:
- **api**: Any OpenAI-compatible API endpoint (OpenAI, Together, Groq, etc.)
- **vllm**: Local vLLM server or in-process generation.

### Usage Examples

```bash
# Using an OpenAI-compatible API
python reasoning/enhance_traces.py \
    --backend api \
    --api_base "https://api.openai.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --model "gpt-4o" \
    --input_file data/reasoning_traces.jsonl \
    --output_file data/enhanced_traces.jsonl

# Using a local vLLM instance
python reasoning/enhance_traces.py \
    --backend vllm \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --input_file data/reasoning_traces.jsonl \
    --output_file data/enhanced_traces.jsonl \
    --tensor_parallel_size 4
```


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

Train a smaller student model on the reasoning traces using Supervised Fine-Tuning (SFT).

```bash
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3b-Instruct" \
    --dataset_name_or_path "unsloth/OpenMathReasoning-mini" \
    --dataset_split "cot" \
    --response_column "generated_solution" \
    --output_dir "./output/Qwen2.5-3b-Instruct-MATH" \
    --load_in_4bit \
    --max_train_samples 600 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --system_prompt "Reason step by step, and put your final answer within \\boxed{}."
```

### 3. Merge LoRA Weights

Merge the LoRA adapter back into the base model for faster inference and evaluation.

```bash
python train/merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen2.5-3b-Instruct" \
    --adapter_path "./output/Qwen2.5-3b-Instruct-MATH" \
    --output_dir "./output/Qwen2.5-3b-Instruct-MATH-merged"
```

### 4. Evaluate

Use the LM Evaluation Harness to evaluate the fine-tuned model on the MATH500 benchmark.

```bash
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3b-Instruct-MATH-merged \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0 \
    --system_instruction "Reason step by step, and put your final answer within \\boxed{}."
```