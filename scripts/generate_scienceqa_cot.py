#!/usr/bin/env python3
"""Phase 1.1 — Off-policy CoT generation for ScienceQA multimodal distillation.

Greedy generation of teacher CoT (Qwen2.5-VL-7B-Instruct, bf16) for ScienceQA
(derek-thomas/ScienceQA). Produces `data/scienceqa_cot_qwen25_vl_7b.jsonl`.

Consumed downstream by `src/data/data_scienceqa.py` via `build_dataloader_cot`.

Design Features:
----------------
* Few-shot exemplars: fixed exemplar demonstrations covering with-image and
  without-image scientific reasoning. Exemplar indices are excluded from train
  distillation targets.
* Multimodal handling: uses `qwen_vl_utils.process_vision_info()` and `AutoProcessor`.
* Resumable writes: reads pre-existing JSONL and skips already processed (split, idx)
  pairs. Allows transparent resumption after Colab disconnects.
* Student length budget: counts tokens under student processor (Qwen2.5-VL-3B-Instruct).
* Output metrics: generates sidecar `data/scienceqa_cot_stats.json` with accuracy and
  token length percentiles.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
try:
    from PIL import Image
except ImportError:
    Image = None

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datasets
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("qwen-vl-utils is required. Install via `pip install qwen-vl-utils`.")

from src.data.data_scienceqa import (
    ANSWER_LETTERS,
    HASH_MARKER,
    SYSTEM_PROMPT,
    answer_index_to_letter,
    format_choices,
)


DEFAULT_TEACHER_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_STUDENT_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_PATH = "data/scienceqa_cot_qwen25_vl_7b.jsonl"
DEFAULT_STATS_PATH = "data/scienceqa_cot_stats.json"
DEFAULT_IMAGES_DIR = "data/scienceqa_images"

# Few-shot exemplar indices in ScienceQA train split
# (Exemplar 0: natural science with image; Exemplar 1: social/language without image)
FEW_SHOT_INDICES = (12, 42)


# ------------------------------------------------------------------
# Resume & IO Helpers
# ------------------------------------------------------------------

def load_done_keys(jsonl_path: str) -> Set[Tuple[str, int]]:
    """Return set of (split, idx) already generated in jsonl_path."""
    if not os.path.isfile(jsonl_path):
        return set()

    done: Set[Tuple[str, int]] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["split"], int(rec["idx"])))
            except Exception:
                continue
    return done


def extract_teacher_answer(text: str) -> Optional[str]:
    """Extract answer option letter from teacher generated completion."""
    # 1. Primary format: #### LETTER
    m = re.search(r"####\s*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()

    # 2. Secondary format: The answer is (LETTER)
    m = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()

    # 3. Format: Answer: LETTER
    m = re.search(r"[Aa]nswer:\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()

    # 4. Fallback: \boxed{LETTER}
    m = re.search(r"\\boxed\{\s*([A-Ea-e])\s*\}", text)
    if m:
        return m.group(1).upper()

    return None


# ------------------------------------------------------------------
# Few-shot Prompt Construction
# ------------------------------------------------------------------

def build_few_shot_messages(
    train_ds: datasets.Dataset,
    few_shot_indices: Tuple[int, ...],
) -> List[dict]:
    """Construct exemplar conversation messages from train split."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for idx in few_shot_indices:
        ex = train_ds[idx]
        question = ex["question"]
        choices = ex["choices"]
        answer_letter = answer_index_to_letter(ex["answer"])

        lecture = ex.get("lecture", "") or ""
        solution = ex.get("solution", "") or ""
        reasoning = f"{lecture}\n{solution}".strip()

        user_content = []
        if ex.get("image") is not None and isinstance(ex["image"], Image.Image):
            user_content.append({"type": "image", "image": ex["image"]})

        prompt_text = f"Question: {question}\n\nChoices:\n{format_choices(choices)}"
        if ex.get("hint"):
            prompt_text = f"Hint: {ex['hint']}\n\n{prompt_text}"
        user_content.append({"type": "text", "text": prompt_text})

        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": f"{reasoning}\n{HASH_MARKER} {answer_letter}",
        })

    return messages


# ------------------------------------------------------------------
# Single Generation Step
# ------------------------------------------------------------------

def generate_one(
    split: str,
    idx: int,
    example: dict,
    few_shot_messages: List[dict],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    student_processor: AutoProcessor,
    device: torch.device,
    max_new_tokens: int = 512,
    images_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate CoT for a single ScienceQA example."""
    question = example["question"]
    choices = example["choices"]
    answer_gold_idx = example["answer"]
    answer_gold_letter = answer_index_to_letter(answer_gold_idx)
    raw_image = example.get("image", None)
    hint = example.get("hint", "") or ""

    has_image = raw_image is not None and isinstance(raw_image, Image.Image)

    # Save image to disk if images_dir is specified
    image_path = None
    if has_image and images_dir:
        image_filename = f"{split}_{idx:06d}.png"
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            raw_image.save(image_path)

    # Build prompt messages: few-shot exemplars + current example
    messages = list(few_shot_messages)

    current_user_content = []
    if has_image:
        current_user_content.append({"type": "image", "image": raw_image})
    prompt_text = f"Question: {question}\n\nChoices:\n{format_choices(choices)}"
    if hint:
        prompt_text = f"Hint: {hint}\n\n{prompt_text}"
    current_user_content.append({"type": "text", "text": prompt_text})
    messages.append({"role": "user", "content": current_user_content})

    # Prepare inputs for generation
    prompt_full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Greedy generation (T=0)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Extract only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0, prompt_len:]
    teacher_full_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()

    # Parse extracted answer
    extracted_answer = extract_teacher_answer(teacher_full_text)
    separator_found = HASH_MARKER in teacher_full_text
    is_teacher_correct = (extracted_answer == answer_gold_letter) if extracted_answer else False

    # Count tokens under student tokenizer for length budgeting
    student_tokens = len(
        student_processor.tokenizer.encode(teacher_full_text, add_special_tokens=False)
    )

    return {
        "split": split,
        "idx": idx,
        "question": question,
        "choices": choices,
        "answer_gold": answer_gold_idx,
        "answer_gold_letter": answer_gold_letter,
        "has_image": has_image,
        "hint": hint,
        "lecture": example.get("lecture", "") or "",
        "solution": example.get("solution", "") or "",
        "teacher_full_text": teacher_full_text,
        "teacher_answer_letter": extracted_answer,
        "is_teacher_correct": is_teacher_correct,
        "separator_found": separator_found,
        "student_token_count": student_tokens,
        "image_path": image_path,
    }


# ------------------------------------------------------------------
# Statistics Calculation
# ------------------------------------------------------------------

def compute_stats(jsonl_path: str) -> Dict[str, Any]:
    """Compute aggregate accuracy, separator adherence, and length percentiles."""
    by_split: Dict[str, List[Dict[str, Any]]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_split.setdefault(r["split"], []).append(r)

    stats: Dict[str, Any] = {}
    import numpy as np

    for s, rows in by_split.items():
        n = len(rows)
        if n == 0:
            continue
        n_correct = sum(1 for r in rows if r.get("is_teacher_correct", False))
        n_sep = sum(1 for r in rows if r.get("separator_found", False))

        # Breakdown with-image vs without-image
        rows_img = [r for r in rows if r.get("has_image", False)]
        rows_noimg = [r for r in rows if not r.get("has_image", False)]

        acc_img = (
            sum(1 for r in rows_img if r.get("is_teacher_correct", False)) / len(rows_img)
            if rows_img else None
        )
        acc_noimg = (
            sum(1 for r in rows_noimg if r.get("is_teacher_correct", False)) / len(rows_noimg)
            if rows_noimg else None
        )

        lengths = [r.get("student_token_count", 0) for r in rows]
        p50 = float(np.percentile(lengths, 50)) if lengths else 0.0
        p90 = float(np.percentile(lengths, 90)) if lengths else 0.0
        p95 = float(np.percentile(lengths, 95)) if lengths else 0.0
        p99 = float(np.percentile(lengths, 99)) if lengths else 0.0

        stats[s] = {
            "n_total": n,
            "accuracy_overall": n_correct / n,
            "accuracy_with_image": acc_img,
            "accuracy_without_image": acc_noimg,
            "separator_adherence": n_sep / n,
            "n_with_image": len(rows_img),
            "n_without_image": len(rows_noimg),
            "token_length_p50": p50,
            "token_length_p90": p90,
            "token_length_p95": p95,
            "token_length_p99": p99,
        }

    return stats


# ------------------------------------------------------------------
# Main Execution Orchestrator
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ScienceQA teacher CoT")
    parser.add_argument("--teacher-name", type=str, default=DEFAULT_TEACHER_NAME)
    parser.add_argument("--student-name", type=str, default=DEFAULT_STUDENT_NAME)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats-path", type=str, default=DEFAULT_STATS_PATH)
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-per-split", type=int, default=None,
                        help="Limit examples per split (for testing/piloting)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Dataset splits to generate (e.g. train test)")
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="Save PIL images to disk for fast offline loading")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Processors
    print(f"Loading teacher processor: {args.teacher_name}")
    processor = AutoProcessor.from_pretrained(
        args.teacher_name,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    print(f"Loading student processor: {args.student_name}")
    student_processor = AutoProcessor.from_pretrained(args.student_name)

    # 2. Load Teacher Model (bf16, eval, frozen)
    print(f"Loading teacher model: {args.teacher_name} in bf16 ...")
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.teacher_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Load ScienceQA Splits
    print("Loading ScienceQA from HuggingFace (derek-thomas/ScienceQA) ...")
    loaded_splits = {}
    for s in args.splits:
        loaded_splits[s] = datasets.load_dataset("derek-thomas/ScienceQA", split=s)
        print(f"  Split {s!r}: {len(loaded_splits[s])} examples")

    train_ds = loaded_splits.get("train", datasets.load_dataset("derek-thomas/ScienceQA", split="train"))

    # 4. Build Few-Shot Messages
    print(f"Building few-shot messages using exemplars: {FEW_SHOT_INDICES}")
    few_shot_messages = build_few_shot_messages(train_ds, FEW_SHOT_INDICES)

    # 5. Check already completed records (resume)
    done_keys = load_done_keys(args.output_path)
    if done_keys:
        print(f"Found {len(done_keys)} completed records in {args.output_path}. Resuming...")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.save_images and args.images_dir:
        os.makedirs(args.images_dir, exist_ok=True)

    img_dir_param = args.images_dir if args.save_images else None

    # 6. Generation Loop
    out_fh = open(args.output_path, "a", encoding="utf-8")
    t0 = time.time()
    n_generated = 0
    n_correct = 0

    try:
        for split_name in args.splits:
            ds = loaded_splits[split_name]
            total_in_split = len(ds)
            print(f"\n>>> Starting split: {split_name} ({total_in_split} items)")

            for idx in range(total_in_split):
                if args.max_per_split and idx >= args.max_per_split:
                    print(f"Reached max_per_split={args.max_per_split} for {split_name}")
                    break

                # Skip few-shot exemplars on train split to prevent data leakage
                if split_name == "train" and idx in FEW_SHOT_INDICES:
                    continue

                if (split_name, idx) in done_keys:
                    continue

                try:
                    rec = generate_one(
                        split=split_name,
                        idx=idx,
                        example=ds[idx],
                        few_shot_messages=few_shot_messages,
                        model=teacher,
                        processor=processor,
                        student_processor=student_processor,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        images_dir=img_dir_param,
                    )
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done_keys.add((split_name, idx))
                    n_generated += 1
                    n_correct += int(rec["is_teacher_correct"])

                    if n_generated % 25 == 0:
                        out_fh.flush()
                        elapsed = time.time() - t0
                        s_per_it = elapsed / n_generated
                        acc = (n_correct / n_generated) * 100.0
                        print(
                            f"[{split_name} {idx:>5d}/{total_in_split}] "
                            f"done={n_generated}  acc={acc:.1f}%  "
                            f"speed={s_per_it:.2f}s/ex"
                        )
                except Exception as ex:
                    print(f"[ERROR] failed on {split_name} idx {idx}: {ex}")
                    continue

    finally:
        out_fh.flush()
        out_fh.close()

    print(f"\nGeneration pass complete. Total newly generated: {n_generated}")

    # 7. Compute and save stats
    stats = compute_stats(args.output_path)
    os.makedirs(os.path.dirname(args.stats_path) or ".", exist_ok=True)
    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {args.stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
