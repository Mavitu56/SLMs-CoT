# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge Distillation (KD) experiment comparing Forward KL (FKL) vs Reverse KL (RKL) vs CE baseline on autoregressive LLMs. Measures token-level entropy H(t), ECE, and KL divergence Teacher→Student.

- **Teacher**: Qwen/Qwen2.5-7B-Instruct (frozen, 4-bit by default)
- **Student**: Qwen/Qwen2.5-1.5B-Instruct (bf16)
- **Primary dataset**: databricks/databricks-dolly-15k
- **Secondary dataset**: cais/mmlu (evaluation only)
- **Active branch**: `cot-teacher-jsonl` — extends Article I (Dolly KD) with GSM8K Chain-of-Thought

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run training with a YAML config
python scripts/run.py --config configs/dolly/dolly_fkl_T2_seed42.yaml

# Override output directory
python scripts/run.py --config configs/dolly/dolly_fkl_T2_seed42.yaml --output-dir results/run_name

# Probabilistic evaluation (single checkpoint)
python scripts/run_analysis.py \
  --config configs/dolly/dolly_fkl_T2_seed42.yaml \
  --checkpoint checkpoints/fkl_T2_seed42/final \
  --label FKL

# Compare multiple checkpoints
python scripts/run_analysis.py \
  --config configs/dolly/dolly_fkl_T2_seed42.yaml \
  --checkpoints CE=checkpoints/ce_T1_seed42/final FKL=checkpoints/fkl_T2_seed42/final RKL=checkpoints/rkl_T2_seed42/final \
  --output-dir analysis_output

# Run all analytical tests
pytest tests/ -v

# Run a single test file
pytest tests/test_losses_numeric.py -v
```

## Architecture

```
scripts/run.py          → entry point; loads YAML, runs sanity, calls src/training/train_manual.py::train()
scripts/run_analysis.py → evaluation entry point; calls src/evaluation/evaluate_probabilistic.py::evaluate_model()
src/
  data/
    data_gsm8k.py       → GSM8K loader; BASE for CoT adaptation; region segmentation (prompt/reasoning/answer)
    data_dolly.py       → Dolly-15k loader; masks prompt tokens in labels (labels[:prompt_len]=-100)
    data_mmlu.py        → MMLU eval loader (MCQ, no shuffle)
  losses/
    losses_kd.py        → IMMUTABLE — all KD loss functions (see below)
  training/
    train_manual.py     → training loop, optimizer, scheduler, checkpointing
    sanity.py           → 11 pre-training checks (teacher frozen, KL≈0, masking, etc.)
    utils_seed.py       → set_seed() — Python/NumPy/PyTorch/CUDA
  evaluation/
    analysis_metrics.py         → token entropy, ECE (10-bin equal-width), KL per position, NLL, perplexity
    evaluate_probabilistic.py   → full eval pass; per-region breakdown; accumulates confidence/correctness for ECE
    evaluate_generation.py      → ROUGE-L on Dolly; MMLU exact-match accuracy
  visualization/
    plotting_utils.py   → bar charts, KL-per-position curves, entropy/maxprob curves
configs/
  dolly/                → 21 YAML files (3 CE + 9 FKL + 9 RKL) for Dolly experiment
tests/
  test_losses_numeric.py → 6 analytical tests on synthetic tensors (all passing)
```

## Critical Rules

### `src/losses_kd.py` is IMMUTABLE
All six functions (`shift_for_causal_lm`, `compute_ce`, `compute_kd_forward_kl`, `compute_kd_reverse_kl`, `compute_total_loss`, and helpers) are validated against analytical tests. **Do not modify this file.**

### Loss Formulas (must not be altered)
| Mode | Formula |
|---|---|
| `ce_only` | `L = L_CE` |
| `fkl` | `L = α·L_CE + (1−α)·T²·L_FKL` — T² applied in `compute_total_loss`, NOT inside `compute_kd_forward_kl` |
| `rkl` | `L = α·L_CE + (1−α)·L_RKL` — no T, no T² |

### Metric Invariants
- All metrics use **T=1 (raw logits)**, computed in eval mode, only over response region tokens.
- ECE uses **10-bin equal-width** on [0,1]. Do not use equal-mass bins.
- KL Teacher→Student metric is separate from the training loss — both always use T=1.

### Region IDs (data_gsm8k.py)
- `REGION_PROMPT=0`, `REGION_REASONING=1`, `REGION_ANSWER=2`
- Segmented by detecting `####` separator in the teacher-generated text.

### Normalization
- All losses normalize by **number of valid tokens |M|**, not by sequence or batch.

## Implementation Order (do not skip steps)

The project follows a strict ordered plan. Current status: **Phase 1.0 COMPLETE → Phase 1.1 in progress**.

1. ✅ Phase 1.0 — Pilot CoT generation (100 examples, GO decision)
2. Phase 1.1 — Generate full GSM8K JSONL (`data/gsm8k_cot_qwen25_7b.jsonl`, 8,788 lines) via `scripts/generate_full_cot.py`
3. Phase 2 — Adapt `src/data_gsm8k.py` to load external JSONL; add CoT metrics; add tests
4. Phase 4 — Run 21 configs on GSM8K-CoT (CE seed42 sanity run first)
5. Phase 5 — Multi-metric analysis

Full plan with gate criteria lives in `Planejamento_KD_v2_2.md`.

## Config Schema

All runs are driven by YAML configs in `configs/`. Required fields:

```yaml
teacher_name / student_name / teacher_load_mode / student_dtype
dataset: dolly | gsm8k | mmlu
kd_mode: ce_only | fkl | rkl
temperature: 1 | 2 | 4          # ignored for rkl
alpha: 0.5                       # fixed
seed: 42 | 123 | 7
max_length: 512                  # 1024 for GSM8K-CoT (Decision D1)
batch_size: 2 / grad_accum_steps: 4   # effective batch = 8
lr: 2.0e-5 / scheduler: cosine / warmup_ratio: 0.1 / weight_decay: 0.05
```

## GSM8K CoT JSONL Schema

Output of `scripts/generate_full_cot.py` (one JSON object per line):

```json
{
  "split": "train" | "test",
  "idx": int,
  "question": str,
  "answer_gold": str,
  "teacher_full_text": str,
  "extracted_answer": str | null,
  "is_teacher_correct": bool,
  "separator_found": bool,
  "total_len_tokens": int
}
```

Key constants in the generation scripts (must remain consistent across phases):
- `max_length: 1024` (Decision D1 from Phase 1.0 pilot)
- `filter_teacher_wrong: false` (Decision D3 — 92% accuracy, filtering not warranted)
- Few-shot indices: `{0, 1, 2, 3}` (excluded from train pool → 7,469 train + 1,319 test)

## Documentation Files

| File | Purpose |
|---|---|
| `Planejamento_KD_v2_2.md` | Master operational plan with phase gates, changelogs, decisions |
| `Pre0_Artigo_II_Consolidado(1).md` | Bibliographic review (33 papers), gap analysis, 10 alerts (A11–A20) |
| `outputs/pilot/pilot_report.md` | Phase 1.0 results — gate criteria, length distribution, GO decision |
| `Prompt_Inicial_Claude_Code.md` | Session conventions (theory docstrings, inventory-first workflow) |
