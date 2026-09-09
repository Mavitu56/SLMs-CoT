"""ScienceQA multiple-choice accuracy and generation evaluation.

Computes:
1. Exact match accuracy on ScienceQA multiple choice answers (letters A-E).
2. Format adherence rate (% of outputs containing the `####` delimiter).
3. Modality segmentation:
   - Accuracy on examples WITH images
   - Accuracy on examples WITHOUT images (text-only)
4. Subject-level breakdown (Natural Science, Social Science, Language Science).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
try:
    from PIL import Image
except ImportError:
    Image = None
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_scienceqa import (
    ANSWER_LETTERS,
    HASH_MARKER,
    SYSTEM_PROMPT,
    answer_index_to_letter,
    build_user_message,
)


def extract_scienceqa_answer(text: str) -> Optional[str]:
    """Extract answer letter from model generated completion.

    Searches in order of strictness:
    1. #### <LETTER>
    2. The answer is (<LETTER>)
    3. Answer: <LETTER>
    4. \\boxed{<LETTER>}
    5. Final isolated letter at end of text
    """
    # 1. Primary format: #### LETTER
    m = re.search(r"####\s*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()

    # 2. Secondary format: "the answer is (LETTER)"
    m = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()

    # 3. Format: "Answer: LETTER"
    m = re.search(r"[Aa]nswer:\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()

    # 4. Fallback: \boxed{LETTER}
    m = re.search(r"\\boxed\{\s*([A-Ea-e])\s*\}", text)
    if m:
        return m.group(1).upper()

    # 5. Last token letter (e.g. "... Therefore, (B)")
    m = re.findall(r"\b([A-Ea-e])\b", text)
    if m:
        return m[-1].upper()

    return None


@torch.no_grad()
def evaluate_scienceqa_accuracy(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    records: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run greedy evaluation over records and compute segmented accuracy metrics."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total = 0
    correct = 0
    separator_count = 0

    total_with_img = 0
    correct_with_img = 0

    total_no_img = 0
    correct_no_img = 0

    subject_stats: Dict[str, Dict[str, int]] = {}
    detailed_outputs = []

    for r in tqdm(records, desc="Evaluating ScienceQA"):
        question = r["question"]
        choices = r["choices"]

        if "answer_gold_letter" in r:
            gold_letter = r["answer_gold_letter"].strip().upper()
        elif "answer_gold" in r:
            gold_letter = answer_index_to_letter(r["answer_gold"])
        elif "answer" in r:
            ans_val = r["answer"]
            gold_letter = answer_index_to_letter(ans_val) if isinstance(ans_val, int) else str(ans_val).upper()
        else:
            continue

        # Load image if present
        image = r.get("image", None)
        if image is None and r.get("image_path") and os.path.isfile(r["image_path"]):
            try:
                image = Image.open(r["image_path"]).convert("RGB")
            except Exception:
                image = None

        has_image = (image is not None and isinstance(image, Image.Image)) or r.get("has_image", False)
        hint = r.get("hint", "") or ""
        subject = r.get("subject", "unknown")

        # Build prompt messages for generation
        user_msg = build_user_message(question, choices, image, hint)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        generated_tokens = output_ids[0, prompt_len:]
        completion = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        pred_letter = extract_scienceqa_answer(completion)
        is_correct = (pred_letter == gold_letter) if pred_letter else False
        has_sep = HASH_MARKER in completion

        total += 1
        if is_correct:
            correct += 1
        if has_sep:
            separator_count += 1

        if has_image:
            total_with_img += 1
            if is_correct:
                correct_with_img += 1
        else:
            total_no_img += 1
            if is_correct:
                correct_no_img += 1

        # Subject breakdown
        sub_entry = subject_stats.setdefault(subject, {"total": 0, "correct": 0})
        sub_entry["total"] += 1
        if is_correct:
            sub_entry["correct"] += 1

        detailed_outputs.append({
            "idx": r.get("idx"),
            "gold": gold_letter,
            "pred": pred_letter,
            "is_correct": is_correct,
            "has_image": has_image,
            "subject": subject,
            "completion": completion,
        })

    acc_overall = correct / max(total, 1)
    acc_img = correct_with_img / max(total_with_img, 1) if total_with_img > 0 else None
    acc_noimg = correct_no_img / max(total_no_img, 1) if total_no_img > 0 else None
    sep_rate = separator_count / max(total, 1)

    subjects_summary = {
        sub: {
            "accuracy": sub_data["correct"] / sub_data["total"],
            "count": sub_data["total"],
        }
        for sub, sub_data in subject_stats.items()
        if sub_data["total"] > 0
    }

    results = {
        "n_total": total,
        "n_correct": correct,
        "accuracy_overall": acc_overall,
        "n_with_image": total_with_img,
        "accuracy_with_image": acc_img,
        "n_without_image": total_no_img,
        "accuracy_without_image": acc_noimg,
        "separator_adherence": sep_rate,
        "by_subject": subjects_summary,
        "detailed_outputs": detailed_outputs,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ScienceQA accuracy")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained student checkpoint or HF model name")
    parser.add_argument("--processor-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--jsonl-path", type=str, default="data/scienceqa_cot_qwen25_vl_7b.jsonl")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading processor: {args.processor_name}")
    processor = AutoProcessor.from_pretrained(args.processor_name)

    print(f"Loading model: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load test records
    records = []
    if os.path.isfile(args.jsonl_path):
        with open(args.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") == args.split:
                    records.append(r)
    else:
        # Load from HuggingFace
        import datasets
        ds = datasets.load_dataset("derek-thomas/ScienceQA", split=args.split)
        for idx, ex in enumerate(ds):
            records.append({
                "idx": idx,
                "split": args.split,
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": ex["answer"],
                "image": ex.get("image"),
                "hint": ex.get("hint", ""),
                "subject": ex.get("subject", ""),
            })

    if args.max_examples:
        records = records[:args.max_examples]

    print(f"Loaded {len(records)} examples for evaluation on split {args.split!r}")
    results = evaluate_scienceqa_accuracy(
        model=model,
        processor=processor,
        records=records,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print("\n" + "=" * 60)
    print("SCIENCEQA EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy:       {results['accuracy_overall']*100:.2f}% ({results['n_correct']}/{results['n_total']})")
    if results['accuracy_with_image'] is not None:
        print(f"Accuracy (With Image):    {results['accuracy_with_image']*100:.2f}% ({results['n_with_image']} items)")
    if results['accuracy_without_image'] is not None:
        print(f"Accuracy (Without Image): {results['accuracy_without_image']*100:.2f}% ({results['n_without_image']} items)")
    print(f"Separator Adherence:    {results['separator_adherence']*100:.2f}%")
    print("=" * 60)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        # Exclude detailed outputs from summary json to keep file compact
        summary = {k: v for k, v in results.items() if k != "detailed_outputs"}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
