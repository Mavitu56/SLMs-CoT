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


def _prepare_record(r: Dict[str, Any], processor: AutoProcessor) -> Tuple[
    str, Optional[list], str, bool, str
]:
    """Prepare a single record for evaluation. Returns (text, images, gold_letter, has_image, subject)."""
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
        return None

    image = r.get("image", None)
    if image is None and r.get("image_path") and os.path.isfile(r["image_path"]):
        try:
            image = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            image = None

    has_image = (image is not None and isinstance(image, Image.Image)) or r.get("has_image", False)
    hint = r.get("hint", "") or ""
    subject = r.get("subject", "unknown")

    user_msg = build_user_message(question, choices, image, hint)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_msg,
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    return text, image_inputs, gold_letter, has_image, subject


@torch.no_grad()
def evaluate_scienceqa_accuracy(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    records: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run batched greedy evaluation over records and compute segmented accuracy metrics.

    Uses left-padded batched generation for much faster throughput on high-VRAM GPUs.
    With batch_size=16 on A100 80GB, evaluation of 4,241 test examples takes ~15-25 min
    instead of ~2+ hours with batch_size=1.
    """
    import time

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

    t_start = time.time()
    n_batches = (len(records) + batch_size - 1) // batch_size

    print(f"\n  Avaliando {len(records)} exemplos em {n_batches} batches (bs={batch_size})...", flush=True)

    # Save and set left padding for batched generation
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    try:
        for batch_idx in range(0, len(records), batch_size):
            batch_records = records[batch_idx:batch_idx + batch_size]

            # Prepare all items in batch
            batch_texts = []
            batch_images = []
            batch_meta = []  # (gold_letter, has_image, subject, record)

            for r in batch_records:
                result = _prepare_record(r, processor)
                if result is None:
                    continue
                text, image_inputs, gold_letter, has_image, subject = result
                batch_texts.append(text)
                if image_inputs:
                    batch_images.extend(image_inputs)
                batch_meta.append((gold_letter, has_image, subject, r))

            if not batch_texts:
                continue

            # Tokenize and generate
            inputs = processor(
                text=batch_texts,
                images=batch_images if batch_images else None,
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

            # Process each output in the batch
            for i, (gold_letter, has_image, subject, r) in enumerate(batch_meta):
                generated_tokens = output_ids[i, prompt_len:]
                completion = processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

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

            # Free memory
            del inputs, output_ids
            torch.cuda.empty_cache()

            # Progress logging
            elapsed = time.time() - t_start
            current_batch = batch_idx // batch_size + 1
            s_per_ex = elapsed / max(total, 1)
            remaining = (len(records) - total) * s_per_ex
            eta_str = f"{remaining/60:.1f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"
            acc_pct = correct / max(total, 1) * 100
            pct = (total / len(records)) * 100.0

            if current_batch <= 3 or current_batch % 5 == 0 or current_batch == n_batches:
                acc_img_pct = (correct_with_img / max(total_with_img, 1)) * 100.0 if total_with_img > 0 else 0.0
                acc_noimg_pct = (correct_no_img / max(total_no_img, 1)) * 100.0 if total_no_img > 0 else 0.0
                print(
                    f"  [{current_batch:>3d}/{n_batches} ({pct:>5.1f}%)] "
                    f"avaliados={total:>4d}/{len(records)} | "
                    f"acurácia={acc_pct:.1f}% (img={acc_img_pct:.1f}%, texto={acc_noimg_pct:.1f}%) | "
                    f"vel={s_per_ex:.2f}s/ex | ETA={eta_str}",
                    flush=True,
                )
    finally:
        processor.tokenizer.padding_side = original_padding_side

    elapsed_total = time.time() - t_start
    print(f"\n  ✓ Avaliação concluída em {elapsed_total/60:.1f} min ({total} exemplos)", flush=True)

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
    parser.add_argument("--max-new-tokens", type=int, default=384,
                        help="Max tokens to generate per example (default: 384).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation. Default: 16.")
    parser.add_argument("--attn-impl", type=str, default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation for the model.")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading processor: {args.processor_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.processor_name)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl

    print(f"Loading model: {args.model_path} (attn={args.attn_impl})", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, **load_kwargs,
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

    # Check if images exist on disk or need HF dataset fallback
    need_hf_images = False
    for r in records[:50]:
        if r.get("has_image") and (not r.get("image_path") or not os.path.isfile(r.get("image_path", ""))):
            need_hf_images = True
            break

    if need_hf_images:
        print("  ℹ️ Imagens locais não encontradas no disco. Vinculando imagens do HuggingFace (derek-thomas/ScienceQA)...", flush=True)
        try:
            import datasets
            ds = datasets.load_dataset("derek-thomas/ScienceQA", split=args.split)
            linked = 0
            for r in records:
                idx = r.get("idx")
                if idx is not None and idx < len(ds) and r.get("has_image"):
                    r["image"] = ds[idx].get("image")
                    linked += 1
            print(f"  ✓ {linked} imagens do HuggingFace vinculadas com sucesso!", flush=True)
        except Exception as e:
            print(f"  ⚠️ Não foi possível carregar imagens do HuggingFace: {e}", flush=True)

    print(f"Loaded {len(records)} examples for evaluation on split {args.split!r}", flush=True)
    results = evaluate_scienceqa_accuracy(
        model=model,
        processor=processor,
        records=records,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
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
