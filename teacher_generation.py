from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

import torch

from config import GenerationConfig
from prompts import (
    build_one_shot_demo,
    build_one_shot_teacher_demo_for_post_cot_gold_rationale,
    build_teacher_cot_prompt,
)


def _get_question(example: Any) -> str:
    if isinstance(example, dict):
        for k in ("question", "input", "prompt", "text"):
            v = example.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(example or "").strip()


def _extract_gold_answer(example: Any) -> str:
    if isinstance(example, dict):
        for key in ("final_answer", "answer", "target", "label", "output"):
            v = example.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        v = example.get("final_answer")
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _normalize_answer(ans: str) -> str:
    return (ans or "").strip().replace("\n", " ").strip().lower()


def _safe_text(x: Any) -> str:
    return str(x or "").strip()


def _batched_generate_continuations(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    prompt_max_length: int,
    gen_cfg: GenerationConfig,
    batch_size: int,
) -> List[str]:
    if not prompts:
        return []

    from tqdm.auto import tqdm

    try:
        model.eval()
    except Exception:
        pass

    batch_size = max(1, int(batch_size or 1))
    outs: List[str] = []

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(prompts), batch_size), desc="Teacher CoT generation", total=total_batches)
    
    for start in pbar:
        chunk = list(prompts[start : start + batch_size])
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(prompt_max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            input_lens = attention_mask.sum(dim=1)
        else:
            input_lens = torch.full((input_ids.size(0),), int(input_ids.size(1)), device=device)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                temperature=float(gen_cfg.temperature),
                do_sample=bool(gen_cfg.do_sample),
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                pad_token_id=pad_token_id,
                repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
            )

        for i in range(int(generated.size(0))):
            in_len = int(input_lens[i].item())
            cont_ids = generated[i][in_len:]
            outs.append(tokenizer.decode(cont_ids, skip_special_tokens=True).strip())

    return outs


def _clean_teacher_reasoning(text: str) -> str:
    """Remove prompt contamination from teacher-generated reasoning.
    
    Teacher may generate garbage when prompt is truncated:
    - May include partial question text at the start
    - May repeat 'Q:', 'A:', 'Let's think step by step' markers
    
    This function cleans up the reasoning to only keep actual reasoning content.
    """
    if not text:
        return ""
    
    # If text starts with ### REASONING: marker, skip it
    m = re.search(r"^###\s*REASONING\s*:\s*", text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():].strip()
    
    # Remove any "Q:" or "A:" markers at the beginning (indicates contamination)
    # Keep removing until we get to clean reasoning content
    while True:
        text = text.strip()
        if not text:
            break
        
        # Check for common contamination patterns at the start
        patterns_to_remove = [
            r"^Q:\s*[^\n]*\n?",  # Q: ... question text
            r"^A:\s*[^\n]*\n?",  # A: ... 
            r"^Let's think step by step\.?\s*\n?",
            r"^###\s*REASONING\s*:\s*",  # Repeated marker
        ]
        
        removed_any = False
        for pat in patterns_to_remove:
            m = re.match(pat, text, flags=re.IGNORECASE)
            if m:
                text = text[m.end():].strip()
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return text.strip()


def _parse_teacher_completion(
    completion: str,
    *,
    post_cot: bool,
    gold_answer: str,
) -> Tuple[str, str]:
    """Parse teacher completion into (reasoning, answer).

    Expected formats:
    - pre-CoT:  <reasoning>\n### FINAL_ANSWER: <answer>
    - post-CoT: <answer>\n### REASONING:\n<reasoning>

    Parsing is best-effort; if missing markers, we fall back conservatively.
    Applies _clean_teacher_reasoning to remove prompt contamination.
    """

    text = (completion or "").strip()
    if not text:
        return "", (gold_answer or "").strip()

    if post_cot:
        m = re.search(r"###\s*REASONING\s*:\s*", text, flags=re.IGNORECASE)
        if m:
            ans = text[: m.start()].strip()
            reasoning = _clean_teacher_reasoning(text[m.end() :].strip())
            if not ans and gold_answer:
                ans = gold_answer.strip()
            return reasoning, ans
        # No marker: treat first line as answer, remainder as reasoning.
        lines = text.splitlines()
        ans = (lines[0] if lines else "").strip()
        reasoning = _clean_teacher_reasoning("\n".join(lines[1:]).strip()) if len(lines) > 1 else ""
        if not ans and gold_answer:
            ans = gold_answer.strip()
        return reasoning, ans

    m = re.search(r"###\s*FINAL_ANSWER\s*:\s*", text, flags=re.IGNORECASE)
    if m:
        reasoning = _clean_teacher_reasoning(text[: m.start()].strip())
        ans = text[m.end() :].strip()
        if not ans and gold_answer:
            ans = gold_answer.strip()
        return reasoning, ans

    # No marker: if we have a gold answer, keep completion as reasoning.
    if gold_answer:
        return _clean_teacher_reasoning(text), gold_answer.strip()

    # Otherwise, heuristic: last non-empty line as answer.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return "", lines[0]
    return _clean_teacher_reasoning("\n".join(lines[:-1]).strip()), lines[-1].strip()


@dataclass
class TeacherCoTGenerationParams:
    granularity_level: int = 0
    granularity_multi_level: bool = False
    one_shot: bool = False
    post_cot: bool = False
    post_cot_gold_rationale: bool = False
    post_cot_use_ig: bool = False
    post_cot_ig_steps: int = 8
    post_cot_ig_top_frac: float = 0.3
    filter_by_gold_answer: bool = False


def generate_teacher_cot_records(
    teacher_model,
    teacher_tokenizer,
    dataset: Any,
    *,
    device: torch.device,
    generation_cfg: GenerationConfig,
    prompt_max_length: int,
    train_limit: Optional[int],
    batch_size: int,
    params: TeacherCoTGenerationParams,
) -> List[Dict[str, Any]]:
    """Generate teacher CoT records in-memory (no disk persistence).

    Returns records compatible with existing downstream consumers (ReasoningAwareDistiller
    and CasCoD pipeline), but never writes to disk.

    Notes:
    - `post_cot_use_ig` is preserved as a flag/field for compatibility, but this
      in-memory generator does not compute integrated gradients; `teacher_reasoning_ig`
      is set to `teacher_reasoning`.
    """

    max_n = None
    try:
        if train_limit is not None:
            max_n = int(train_limit)
    except Exception:
        max_n = None

    # Decide which granularity levels to generate.
    levels: List[int]
    if bool(params.granularity_multi_level) and int(params.granularity_level or 0) > 0:
        levels = list(range(1, int(params.granularity_level) + 1))
    else:
        levels = [int(params.granularity_level or 0)]

    # CRÍTICO: SEMPRE usar one-shot prefix para o teacher saber o formato esperado
    # Sem isso, o teacher não sabe que deve terminar com ### FINAL_ANSWER: <answer>
    # Keep demo short and generic (no dataset-specific leakage).
    if bool(params.post_cot_gold_rationale) and bool(params.post_cot):
        one_shot_prefix = build_one_shot_teacher_demo_for_post_cot_gold_rationale(
            question="If you have 2 apples and buy 3 more, how many apples do you have?",
            answer="5",
            reasoning="Because 2 + 3 = 5.",
        )
    else:
        one_shot_prefix = build_one_shot_demo(
            question="If you have 2 apples and buy 3 more, how many apples do you have?",
            answer="5",
            reasoning="Start with 2 apples. Add 3 more. 2 + 3 = 5 apples total.",
            post_cot=bool(params.post_cot),
        )

    records: List[Dict[str, Any]] = []

    # Generate per level; keep outputs grouped per example.
    # (This mirrors the prior multi-level behavior, but in RAM.)
    questions: List[str] = []
    golds: List[str] = []
    raw_examples: List[Any] = []
    n_total = 0

    for ex in dataset:
        if max_n is not None and n_total >= max_n:
            break
        q = _get_question(ex)
        if not q:
            continue
        questions.append(q)
        golds.append(_extract_gold_answer(ex))
        raw_examples.append(ex)
        n_total += 1

    # Build prompts for each level and batch-generate.
    # We store results in dicts per example.
    per_example: List[Dict[str, Any]] = [
        {
            "question": questions[i],
            "gold_answer": golds[i],
        }
        for i in range(len(questions))
    ]

    for lvl in levels:
        prompts: List[str] = []
        for i, q in enumerate(questions):
            gold = golds[i]
            prompt, _ = build_teacher_cot_prompt(
                q,
                granularity_level=int(lvl),
                post_cot=bool(params.post_cot),
                one_shot_prefix=one_shot_prefix,
                gold_answer=(gold if bool(params.post_cot_gold_rationale) and bool(params.post_cot) else None),
                post_cot_gold_rationale=bool(params.post_cot_gold_rationale),
            )
            prompts.append(prompt)
            if len(levels) == 1:
                per_example[i]["prompt"] = prompt

        completions = _batched_generate_continuations(
            teacher_model,
            teacher_tokenizer,
            prompts,
            device=device,
            prompt_max_length=int(prompt_max_length),
            gen_cfg=generation_cfg,
            batch_size=int(batch_size),
        )

        for i, comp in enumerate(completions):
            reasoning, ans = _parse_teacher_completion(comp, post_cot=bool(params.post_cot), gold_answer=golds[i])
            
            # DEBUG: mostrar primeiros 5 exemplos para verificar qualidade
            if i < 5:
                print(f"\n[DEBUG Teacher Gen] Example {i}:")
                print(f"  Question: {questions[i][:100]}...")
                print(f"  Gold answer: {golds[i]}")
                print(f"  Completion: {comp[:200]}...")
                print(f"  Parsed reasoning: {reasoning[:100]}..." if reasoning else "  Parsed reasoning: EMPTY")
                print(f"  Parsed answer: {ans}")
                has_marker = "### FINAL_ANSWER" in comp.upper()
                print(f"  Has ### FINAL_ANSWER marker: {has_marker}")
            
            if len(levels) == 1:
                per_example[i]["teacher_reasoning"] = reasoning
                per_example[i]["teacher_answer"] = ans
                per_example[i]["teacher_reasoning_ig"] = reasoning
            else:
                k = str(int(lvl))
                per_example[i].setdefault("granularity_levels", []).append(int(lvl))
                per_example[i].setdefault("prompt_levels", {})[k] = prompts[i]
                per_example[i].setdefault("teacher_reasoning_levels", {})[k] = reasoning
                per_example[i].setdefault("teacher_answer_levels", {})[k] = ans
                per_example[i].setdefault("teacher_reasoning_ig_levels", {})[k] = reasoning

    # Apply filtering at the end (keeps generation counts stable across levels).
    for rec in per_example:
        if bool(params.filter_by_gold_answer):
            gold = _normalize_answer(_safe_text(rec.get("gold_answer")))
            if gold:
                ans = _normalize_answer(_safe_text(rec.get("teacher_answer")))
                if ans and ans != gold:
                    continue
        records.append(rec)

    return records
