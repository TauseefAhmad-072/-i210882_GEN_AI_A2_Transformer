# TrialMatch - criterion-level clinical-trial eligibility screening (GenAI Assignment 2)

Educational screening prototype - **not** a clinical enrollment decision tool.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54B params, 28 layers, GQA 12Q/2KV heads, RoPE, 32k ctx)
fine-tuned with LoRA; draft model for speculative decoding: `Qwen/Qwen2.5-0.5B-Instruct` (same tokenizer).

## Layout
| file | task |
|---|---|
| `src/data.py` | dataset, 37/8/8 patient-grouped split (seed 42), manifest, prompts, output parser |
| `src/finetune.py` | Task 1 LoRA fine-tuning, logs, curves |
| `src/evaluate.py`, `src/metrics.py` | Task 1/2 evaluation (base vs fine-tuned), label + evidence metrics |
| `src/decoding.py` | Task 3 KV-cache decoder + uncached reference, Task 5 speculative decoding |
| `src/attention.py` | Task 4 reference + tiled online-softmax attention |
| `src/bench.py` | benchmarks for tasks 3/4/5 |
| `src/pipeline.py`, `app.py` | Task 6 application |
| `tests/test_core.py` | correctness tests (attention, cache equivalence, speculative equivalence, split, metrics) |

## Setup
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pytest -q                                              # must pass before anything else

## Pipeline
    python -m src.data inspect                             # verify column mapping once
    python -m src.data prepare                             # writes data/{train,val,test}.jsonl + split_manifest.json
    python -m src.evaluate --split val --adapter none      # BEFORE fine-tuning
    python -m src.finetune --out outputs/lora              # GPU strongly recommended (Kaggle/Colab T4 is enough)
    python -m src.evaluate --split val --adapter outputs/lora/best     # AFTER fine-tuning
    python -m src.evaluate --split test --adapter outputs/lora/best --final   # once, at the very end
    python -m src.evaluate --split test --adapter none --final --tag base     # same test cases, base model
    python -m src.bench kv   --adapter outputs/lora/best
    python -m src.bench spec --adapter outputs/lora/best
    python -m src.bench attn
    python app.py                                          # http://127.0.0.1:7860

Test split is never read unless `--final` is passed. Benchmarks use validation prompts only.
