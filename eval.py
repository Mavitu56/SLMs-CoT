from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm

from config import EvidenceBasedConfig, GenerationConfig, ensure_tokenizer_has_pad
from prompts import build_cascod_answer_prompt, build_cascod_rationale_prompt


def _batched_generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    *,
    device: torch.device,
    max_length: int,
    gen_cfg: GenerationConfig,
    max_new_tokens: int,
    pad_token_id: Optional[int],
    batch_size: int,
) -> Tuple[List[str], List[int]]:
    """Generate continuations for a list of prompts in micro-batches.

    Returns:
      - continuations: decoded text of generated tokens *after* the prompt
      - cont_token_lens: generated token counts (output_len - input_len)

    This avoids per-example Python overhead and avoids brittle substring slicing
    by using token-level prompt lengths.
    """

    if not prompts:
        return [], []

    batch_size = max(1, int(batch_size or 1))
    continuations: List[str] = []
    cont_token_lens: List[int] = []

    # Ensure eval mode.
    try:
        model.eval()
    except Exception:
        pass

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        input_lens = None
        if attention_mask is not None:
            input_lens = attention_mask.sum(dim=1)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                temperature=float(gen_cfg.temperature),
                do_sample=bool(gen_cfg.do_sample),
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                pad_token_id=pad_token_id,
                repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
            )

        # Slice continuations using per-row prompt length.
        for row in range(int(outputs.shape[0])):
            in_len = int(input_lens[row].item()) if input_lens is not None else int(input_ids.shape[1])
            out_row = outputs[row]
            cont_ids = out_row[in_len:]
            cont_token_lens.append(int(cont_ids.numel()))
            continuations.append(tokenizer.decode(cont_ids, skip_special_tokens=True).strip())

    return continuations, cont_token_lens


def _stable_example_key(ex: Dict[str, Any]) -> str:
    return str(ex.get("input") or ex.get("prompt") or ex.get("question") or ex.get("text") or "")


def split_bbeh_dataset(ds: HFDataset, *, seed: int, eval_fraction: float = 0.2) -> Tuple[HFDataset, HFDataset]:
    """Deterministic train/eval split for BBEH examples.

    Scientific validity fix (leakage): BBEH has no canonical train split in this
    loader. We therefore create a stable hash-based split so that examples used
    in evaluation are never used in training.
    """

    if not 0.0 < float(eval_fraction) < 1.0:
        raise ValueError("eval_fraction must be in (0,1)")

    import hashlib

    eval_idx = []
    train_idx = []
    for i in range(len(ds)):
        ex = ds[i]
        key = _stable_example_key(ex)
        h = hashlib.sha1((str(seed) + "|" + key).encode("utf-8", errors="ignore")).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        if bucket < eval_fraction:
            eval_idx.append(i)
        else:
            train_idx.append(i)

    # Ensure both sides non-empty when dataset is tiny.
    if not eval_idx and len(ds) > 1:
        eval_idx = [0]
        train_idx = list(range(1, len(ds)))
    if not train_idx and len(ds) > 1:
        train_idx = [0]
        eval_idx = list(range(1, len(ds)))

    return ds.select(train_idx), ds.select(eval_idx)


def _normalize_gsm8k_number(s: str) -> str:
    """Normalize a number string for comparison.
    
    Handles: commas, trailing zeros, dollar signs, percentage signs.
    E.g., "1,234.00" -> "1234", "$50" -> "50", "25%" -> "25"
    """
    if not s:
        return ""
    # Remove common prefixes/suffixes
    s = s.replace("$", "").replace("%", "").replace(",", "").strip()
    # Try to parse as float and normalize
    try:
        val = float(s)
        # If it's a whole number, return as int string
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def extract_gsm8k_answer(text: str) -> str:
    if not text:
        return ""
    
    # Check for ### FINAL_ANSWER: marker first (matches training format)
    if "### FINAL_ANSWER" in text.upper():
        m = re.search(r"###\s*FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            # Extract just the number from the answer
            nums = re.findall(r"[\d,]+\.?\d*", ans)
            if nums:
                return _normalize_gsm8k_number(nums[0])
            return ans.split()[0] if ans else ""
    
    if "####" in text:
        ans = text.split("####")[-1].strip()
        return _normalize_gsm8k_number(ans)
    patterns = [
        r"answer[\s:\-]*([\$\d\.,]+)",
        r"final[\s:\-]*([\$\d\.,]+)",
        r"=[\s]*([\$\d\.,]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return _normalize_gsm8k_number(matches[-1].strip())
    numbers = re.findall(r"\d+\.?\d*", text)
    return _normalize_gsm8k_number(numbers[-1]) if numbers else ""


def extract_gsm8k_answer_first(text: str) -> str:
    """Extract the *first* plausible numeric answer from a completion.

    Used for Post-CoT eval (answer-first). This is intentionally simple and
    only activated when explicitly requested to avoid changing older results.
    """

    if not text:
        return ""

    t = (text or "").strip()
    if not t:
        return ""

    # Prefer explicit patterns if present.
    patterns = [
        r"answer[\s:\-]*([\$\d\.,]+)",
        r"final[\s:\-]*([\$\d\.,]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()

    # Otherwise, take the first number-like span.
    nums = re.findall(r"\d+\.?\d*", t)
    return nums[0] if nums else ""


def bbh_answer_match(generated: str, target: str) -> bool:
    g = (generated or "").lower().strip()
    t = (target or "").lower().strip()
    if g == t:
        return True
    if t and t in g:
        return True
    bool_map = {"true": "yes", "false": "no", "yes": "true", "no": "false"}
    return g in bool_map and bool_map[g] == t


def _extract_choice_letter(text: str, *, choices: str) -> str:
    """Extract a multiple-choice letter from model output.

    `choices` should be a string like "ABCD".
    """

    if not text:
        return ""
    t = (text or "").strip()
    if not t:
        return ""
    first = t[0].upper()
    if first in choices:
        return first

    m = re.search(r"\b([%s])\b" % re.escape(choices), t.upper())
    if m:
        return str(m.group(1)).upper()

    # Common pattern: "Answer: B"
    m = re.search(r"answer\s*[:\-]\s*([%s])" % re.escape(choices), t, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).upper()

    return ""


def _starts_with_answer_like(text: str, *, mode: str) -> bool:
    """Heuristic for 'answer-first' detection.

    Modes:
    - 'numeric': begins with a number (optionally $ or -)
    - 'letter': begins with A/B/C/D (or other supplied letters already extracted upstream)
    - 'bool': begins with yes/no/true/false

    This is intentionally cheap and robust (used as a secondary metric).
    """

    t = (text or "").lstrip()
    if not t:
        return False

    if mode == "numeric":
        return re.match(r"^[\$\-]?\d", t) is not None
    if mode == "letter":
        return re.match(r"^[A-Za-z]", t) is not None
    if mode == "bool":
        return re.match(r"^(yes|no|true|false)\b", t, flags=re.IGNORECASE) is not None
    return False


def _normalize_number_string(x: str) -> str:
    if not x:
        return ""
    s = (x or "").strip()
    s = s.replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    return s


def _extract_last_number_like(text: str) -> str:
    if not text:
        return ""
    # Includes optional $ and -, accepts commas and decimals.
    nums = re.findall(r"[\$\-]?\d[\d,]*\.?\d*", text)
    return nums[-1].strip() if nums else ""


def load_bbeh_task_dataset(task_name: str) -> Tuple[HFDataset, str]:
    """Loads a BBEH task JSON from GitHub (no local caching)."""

    repo_alias_map = {
        "logical_deduction_five_objects": ["bbeh_word_sorting", "bbeh_boolean_expressions"],
        "causal_judgement": ["bbeh_causal_understanding"],
        "formal_fallacies": ["bbeh_hyperbaton", "bbeh_boolean_expressions"],
    }
    normalized = (task_name or "").strip()
    candidates = repo_alias_map.get(normalized, [normalized])

    last_err = None
    for directory in candidates:
        try:
            dataset = _load_bbeh_task_from_repo(directory)
            if dataset is not None:
                return dataset, directory
        except Exception as err:
            last_err = err
            continue
    raise RuntimeError(f"No foi possvel carregar task {task_name}: {last_err}")


def _load_bbeh_task_from_repo(directory_name: str) -> HFDataset:
    from datasets import Dataset as HFDataset

    base_url = "https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks"

    url = f"{base_url}/{directory_name}/task.json"
    print(f" Baixando task '{directory_name}' do BBEH...")
    response = requests.get(url, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"No foi possvel baixar {directory_name} ({response.status_code}).")
    payload = response.json()

    if isinstance(payload, dict) and "examples" in payload:
        examples = payload["examples"]
    elif isinstance(payload, list):
        examples = payload
    else:
        raise ValueError(f"Formato inesperado para a task {directory_name}.")

    if not examples:
        raise ValueError(f"Task {directory_name} sem exemplos.")

    return HFDataset.from_list(examples)


class StandardizedEvaluator:
    """Essential evaluation suite for H1 (GSM8K + BBH) + efficiency.

    Additional domains can be enabled via flags in run_experiment.py.
    """

    def __init__(self, config: EvidenceBasedConfig):
        self.config = config

    def evaluate(
        self,
        model,
        tokenizer,
        seed: int,
        eval_gsm8k: bool = True,
        eval_bbh: bool = True,
        eval_obqa: bool = False,
        eval_efficiency: bool = True,
        use_cot_prompt: bool = True,
        answer_first_eval: bool = False,
        cascod_two_stage: bool = False,
        generation_cfg: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        generation_cfg = generation_cfg or self.config.eval_generation
        torch.manual_seed(seed)

        results: Dict[str, Any] = {
            "metadata": {
                "seed": seed,
                "use_cot_prompt": bool(use_cot_prompt),
                "answer_first_eval": bool(answer_first_eval),
                "cascod_two_stage": bool(cascod_two_stage),
                "generation": generation_cfg.to_jsonable(),
            }
        }

        if eval_gsm8k:
            results["gsm8k"] = self._eval_gsm8k(
                model,
                tokenizer,
                seed,
                use_cot_prompt,
                generation_cfg,
                answer_first_eval=bool(answer_first_eval),
                cascod_two_stage=bool(cascod_two_stage),
            )
        if eval_bbh:
            results["bbh"] = self._eval_bbh(model, tokenizer, seed, generation_cfg, cascod_two_stage=bool(cascod_two_stage))
        if eval_obqa:
            results["obqa"] = self._eval_obqa(model, tokenizer, seed, generation_cfg)
        if eval_efficiency:
            # Secondary metric: not part of hypothesis test by default.
            results["efficiency"] = self._eval_efficiency(model, tokenizer, seed, generation_cfg)

        # Secondary metrics (cheap): aggregate across the enabled tasks.
        results["secondary_metrics"] = self._aggregate_secondary_metrics(results)

        # Scientific validity fix: keep a primary score based on task performance.
        results["primary_score"] = self._primary_score(results)
        # Backward-compatible aggregate (still recorded), but treated as secondary.
        results["overall_score"] = self._overall_score(results)
        return results

    def _aggregate_secondary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate cheap, run-level metrics across tasks.

        This intentionally does not affect primary score/hypothesis testing.
        """

        total_items = 0
        total_generated_tokens = 0.0
        total_answer_first = 0.0

        def _accumulate(section: Dict[str, Any], *, key_total: str = "total") -> None:
            nonlocal total_items, total_generated_tokens, total_answer_first
            if not section:
                return
            sec = section.get("secondary") or {}
            try:
                n = int(section.get(key_total) or 0)
            except Exception:
                n = 0
            if n <= 0:
                return

            # Values are means at the task level.
            gen_mean = float(sec.get("generated_tokens_mean", 0.0) or 0.0)
            af_rate = float(sec.get("answer_first_rate", 0.0) or 0.0)
            total_items += n
            total_generated_tokens += gen_mean * n
            total_answer_first += af_rate * n

        _accumulate(results.get("gsm8k", {}) or {}, key_total="total")
        _accumulate(results.get("bbh", {}) or {}, key_total="tasks_evaluated")  # BBH 'total' is per-task.
        _accumulate(results.get("obqa", {}) or {}, key_total="total")

        return {
            "answer_first_rate": float(total_answer_first / total_items) if total_items else 0.0,
            "generated_tokens_mean": float(total_generated_tokens / total_items) if total_items else 0.0,
            "items_covered": int(total_items),
        }

    def _primary_score(self, results: Dict[str, Any]) -> float:
        gsm_present = "gsm8k" in results
        bbh_present = "bbh" in results
        gsm = float(results.get("gsm8k", {}).get("accuracy", 0.0))
        bbh = float(results.get("bbh", {}).get("average_accuracy", 0.0))
        if gsm_present and bbh_present:
            return 0.5 * gsm + 0.5 * bbh
        if gsm_present:
            return gsm
        if bbh_present:
            return bbh
        return 0.0

    def _overall_score(self, results: Dict[str, Any]) -> float:
        gsm = float(results.get("gsm8k", {}).get("accuracy", 0.0))
        bbh = float(results.get("bbh", {}).get("average_accuracy", 0.0))
        eff = results.get("efficiency", {})
        t = float(eff.get("inference_speed_seconds", 10.0))
        # Keep same spirit: performance + small efficiency weight.
        speed_score = max(0.0, 1.0 - (t / 5.0))
        return 0.45 * gsm + 0.45 * bbh + 0.10 * speed_score

    def _eval_obqa(self, model, tokenizer, seed: int, gen_cfg: GenerationConfig) -> Dict[str, Any]:
        """OpenBookQA (commonsense) eval-only OOD.

        Scientific controls:
        - Uses the official test split.
        - Deterministic sampling per seed.
        - Exact-match on the multiple-choice letter.
        """

        from datasets import load_dataset

        ds = load_dataset("openbookqa", "main", split="test")
        limit = min(int(self.config.eval_limit_obqa), len(ds))
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), limit, replace=False) if limit < len(ds) else np.arange(len(ds))

        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        correct = 0
        total = 0
        generated_token_lens: List[int] = []
        answer_first: List[int] = []

        # Build prompts once, then batch-generate.
        prompts: List[str] = []
        golds: List[str] = []
        valid_letters_list: List[str] = []

        for idx in indices:
            ex = ds[int(idx)]
            stem = str(ex.get("question_stem", ""))
            choices_obj = ex.get("choices") or {}
            texts = list(choices_obj.get("text") or [])
            labels = list(choices_obj.get("label") or [])
            gold = str(ex.get("answerKey", "")).strip().upper()

            pairs = []
            for lab, txt in zip(labels, texts):
                lab_u = str(lab).strip().upper()
                if lab_u and txt is not None:
                    pairs.append((lab_u, str(txt)))
            if not pairs:
                continue

            pairs = sorted(pairs, key=lambda p: p[0])
            valid_letters = "".join([p[0] for p in pairs])
            lines = [f"Q: {stem}", "Choices:"]
            for lab, txt in pairs:
                lines.append(f"{lab}) {txt}")
            prompt = "\n".join(lines) + "\nAnswer:"

            prompts.append(prompt)
            golds.append(gold)
            valid_letters_list.append(valid_letters)

        batch_size = int(os.environ.get("SLM_EVAL_BATCH_SIZE", "4") or 4)
        conts, cont_lens = _batched_generate_continuations(
            model,
            tokenizer,
            prompts,
            device=self.config.device,
            max_length=int(self.config.max_length),
            gen_cfg=gen_cfg,
            max_new_tokens=min(8, int(gen_cfg.max_new_tokens)),
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            batch_size=batch_size,
        )

        for gen_answer, gold, valid_letters, clen in tqdm(
            list(zip(conts, golds, valid_letters_list, cont_lens)),
            desc="Eval OBQA",
        ):
            generated_token_lens.append(int(clen))
            answer_first.append(1 if _starts_with_answer_like(gen_answer, mode="letter") else 0)
            pred = _extract_choice_letter(gen_answer, choices=valid_letters)
            if pred and gold and pred == gold:
                correct += 1
            total += 1

        sec = {
            "generated_tokens_mean": float(np.mean(generated_token_lens)) if generated_token_lens else 0.0,
            "answer_first_rate": float(np.mean(answer_first)) if answer_first else 0.0,
        }
        return {"accuracy": correct / total if total else 0.0, "correct": int(correct), "total": int(total), "secondary": sec}

    def _eval_gsm8k(
        self,
        model,
        tokenizer,
        seed: int,
        use_cot_prompt: bool,
        gen_cfg: GenerationConfig,
        *,
        answer_first_eval: bool = False,
        cascod_two_stage: bool = False,
    ) -> Dict[str, Any]:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        limit = min(self.config.eval_limit_gsm8k, len(ds))
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), limit, replace=False)

        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        correct = 0
        total = 0
        generated_token_lens: List[int] = []
        cot_token_lens: List[int] = []
        answer_first: List[int] = []
        consistency: List[int] = []
        questions: List[str] = []
        golds: List[str] = []
        for idx in indices:
            ex = ds[int(idx)]
            questions.append(str(ex.get("question", "")))
            golds.append(extract_gsm8k_answer(ex.get("answer", "") or ex.get("output", "")))

        batch_size = int(os.environ.get("SLM_EVAL_BATCH_SIZE", "4") or 4)

        if bool(cascod_two_stage):
            # Stage 1 (q -> r)
            prompts_r = [build_cascod_rationale_prompt(q, granularity_level=0) for q in questions]
            conts_r, lens_r = _batched_generate_continuations(
                model,
                tokenizer,
                prompts_r,
                device=self.config.device,
                max_length=int(self.config.max_length),
                gen_cfg=gen_cfg,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
                batch_size=batch_size,
            )

            # Stage 2 (q,r -> a)
            prompts_a = [build_cascod_answer_prompt(q, r) for q, r in zip(questions, conts_r)]
            conts_a, lens_a = _batched_generate_continuations(
                model,
                tokenizer,
                prompts_a,
                device=self.config.device,
                max_length=int(self.config.max_length),
                gen_cfg=gen_cfg,
                max_new_tokens=min(32, int(gen_cfg.max_new_tokens)),
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
                batch_size=batch_size,
            )

            for rationale, cont_text, gold, r_len, a_len in tqdm(
                list(zip(conts_r, conts_a, golds, lens_r, lens_a)),
                desc="Eval GSM8K",
            ):
                generated_token_lens.append(int(r_len) + int(a_len))
                pred = extract_gsm8k_answer(cont_text)
                answer_first.append(1 if _starts_with_answer_like(cont_text, mode="numeric") else 0)

                if (int(r_len) + int(a_len)) > 0:
                    try:
                        ans_ids = tokenizer(pred or "", add_special_tokens=False).get("input_ids") or []
                        ans_len = int(len(ans_ids))
                    except Exception:
                        ans_len = 0
                    cot_token_lens.append(max(0, int(r_len) + int(a_len) - int(ans_len)))

                r_last = _normalize_number_string(_extract_last_number_like(rationale))
                a_norm = _normalize_number_string(pred)
                if r_last and a_norm:
                    consistency.append(1 if r_last == a_norm else 0)

                if pred == gold and gold != "":
                    correct += 1
                total += 1

        else:
            # Single-pass generation.
            # IMPORTANT: prompt must match training format with one-shot + explicit instruction
            # O one-shot + instrução explícita ensina o modelo o formato esperado
            one_shot_demo = (
                "Example:\n"
                "Q: If you have 2 apples and buy 3 more, how many apples do you have?\n"
                "A: Let's think step by step. After reasoning, conclude with ### FINAL_ANSWER: <number>\n"
                "### REASONING:\n"
                "Start with 2 apples. Add 3 more. 2 + 3 = 5 apples total.\n"
                "### FINAL_ANSWER: 5\n\n"
            )
            if use_cot_prompt:
                prompts = [f"{one_shot_demo}Q: {q}\nA: Let's think step by step. After reasoning, conclude with ### FINAL_ANSWER: <number>\n### REASONING:\n" for q in questions]
            else:
                prompts = [f"Q: {q}\nA:" for q in questions]

            conts, cont_lens = _batched_generate_continuations(
                model,
                tokenizer,
                prompts,
                device=self.config.device,
                max_length=int(self.config.max_length),
                gen_cfg=gen_cfg,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
                batch_size=batch_size,
            )

            # DEBUG: track examples for analysis
            debug_eval = os.environ.get("SLM_DEBUG_EVAL", "1").strip().lower() in {"1", "true", "yes"}
            debug_shown = 0
            
            for cont_text, cont_len, gold in tqdm(
                list(zip(conts, cont_lens, golds)),
                desc="Eval GSM8K",
            ):
                generated_token_lens.append(int(cont_len))

                if (not use_cot_prompt) and bool(answer_first_eval):
                    pred = extract_gsm8k_answer_first(cont_text)
                else:
                    pred = extract_gsm8k_answer(cont_text)

                # DEBUG: mostrar primeiros 10 exemplos para análise
                if debug_eval and debug_shown < 10:
                    is_correct = (pred == gold and gold != "")
                    print(f"\n[DEBUG Eval GSM8K] Example {debug_shown}:")
                    print(f"  Gold: '{gold}'")
                    print(f"  Pred: '{pred}'")
                    print(f"  Match: {is_correct}")
                    print(f"  Generation (first 300 chars): {cont_text[:300]}...")
                    has_marker = "### FINAL_ANSWER" in cont_text.upper()
                    print(f"  Has ### FINAL_ANSWER marker: {has_marker}")
                    debug_shown += 1

                if pred == gold and gold != "":
                    correct += 1
                total += 1

                answer_first.append(1 if _starts_with_answer_like(cont_text, mode="numeric") else 0)

                if use_cot_prompt and int(cont_len) > 0:
                    try:
                        ans_ids = tokenizer(pred or "", add_special_tokens=False).get("input_ids") or []
                        ans_len = int(len(ans_ids))
                    except Exception:
                        ans_len = 0
                    cot_token_lens.append(max(0, int(cont_len) - int(ans_len)))

                if use_cot_prompt:
                    reasoning_text = cont_text
                    if "####" in reasoning_text:
                        reasoning_text = reasoning_text.split("####")[0]
                    else:
                        if pred:
                            pos = reasoning_text.rfind(str(pred))
                            reasoning_text = reasoning_text[:pos] if pos > 0 else reasoning_text
                    r_last = _normalize_number_string(_extract_last_number_like(reasoning_text))
                    a_norm = _normalize_number_string(pred)
                    if r_last and a_norm:
                        consistency.append(1 if r_last == a_norm else 0)

        sec = {
            "generated_tokens_mean": float(np.mean(generated_token_lens)) if generated_token_lens else 0.0,
            "cot_tokens_mean": float(np.mean(cot_token_lens)) if cot_token_lens else 0.0,
            "answer_first_rate": float(np.mean(answer_first)) if answer_first else 0.0,
            "consistency_rate": float(np.mean(consistency)) if consistency else 0.0,
        }
        return {"accuracy": correct / total if total else 0.0, "correct": int(correct), "total": int(total), "secondary": sec}

    def _eval_bbh(self, model, tokenizer, seed: int, gen_cfg: GenerationConfig, *, cascod_two_stage: bool = False) -> Dict[str, Any]:
        task_names = ["logical_deduction_five_objects", "causal_judgement", "formal_fallacies"]
        rng = np.random.RandomState(seed)

        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        all_results: Dict[str, Any] = {}
        total_accuracy = 0.0
        per_task_generated_tokens: List[float] = []
        per_task_answer_first: List[float] = []

        for task in task_names:
            ds, src = load_bbeh_task_dataset(task)
            train_split, eval_split = split_bbeh_dataset(ds, seed=seed, eval_fraction=0.2)
            ds = eval_split
            limit = min(self.config.eval_limit_bbh, len(ds))
            indices = rng.choice(len(ds), limit, replace=False)

            task_correct = 0
            generated_token_lens: List[int] = []
            answer_first: List[int] = []

            inputs_texts: List[str] = []
            targets: List[str] = []
            for idx in indices:
                ex = ds[int(idx)]
                inputs_texts.append(str(ex.get("input", "") or ex.get("prompt", "")))
                targets.append(str(ex.get("target", "") or ex.get("output", "")))

            batch_size = int(os.environ.get("SLM_EVAL_BATCH_SIZE", "4") or 4)

            if bool(cascod_two_stage):
                prompts_r = [build_cascod_rationale_prompt(x, granularity_level=0) for x in inputs_texts]
                conts_r, lens_r = _batched_generate_continuations(
                    model,
                    tokenizer,
                    prompts_r,
                    device=self.config.device,
                    max_length=int(self.config.max_length),
                    gen_cfg=gen_cfg,
                    max_new_tokens=int(gen_cfg.max_new_tokens),
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    batch_size=batch_size,
                )

                prompts_a = [build_cascod_answer_prompt(x, r) for x, r in zip(inputs_texts, conts_r)]
                conts_a, lens_a = _batched_generate_continuations(
                    model,
                    tokenizer,
                    prompts_a,
                    device=self.config.device,
                    max_length=int(self.config.max_length),
                    gen_cfg=gen_cfg,
                    max_new_tokens=min(64, int(gen_cfg.max_new_tokens)),
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    batch_size=batch_size,
                )

                for gen_answer, target, r_len, a_len in tqdm(
                    list(zip(conts_a, targets, lens_r, lens_a)),
                    desc=f"Eval BBH {task}",
                ):
                    generated_token_lens.append(int(r_len) + int(a_len))
                    answer_first.append(
                        1
                        if _starts_with_answer_like(gen_answer, mode="bool")
                        or _starts_with_answer_like(gen_answer, mode="letter")
                        else 0
                    )
                    if bbh_answer_match(gen_answer, target or ""):
                        task_correct += 1

            else:
                prompts = [f"{x}\nAnswer:" for x in inputs_texts]
                conts, cont_lens = _batched_generate_continuations(
                    model,
                    tokenizer,
                    prompts,
                    device=self.config.device,
                    max_length=int(self.config.max_length),
                    gen_cfg=gen_cfg,
                    max_new_tokens=min(64, int(gen_cfg.max_new_tokens)),
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    batch_size=batch_size,
                )

                for gen_answer, target, clen in tqdm(
                    list(zip(conts, targets, cont_lens)),
                    desc=f"Eval BBH {task}",
                ):
                    generated_token_lens.append(int(clen))
                    answer_first.append(
                        1
                        if _starts_with_answer_like(gen_answer, mode="bool")
                        or _starts_with_answer_like(gen_answer, mode="letter")
                        else 0
                    )
                    if bbh_answer_match(gen_answer, target or ""):
                        task_correct += 1

            task_acc = task_correct / limit if limit else 0.0
            all_results[task] = {"accuracy": float(task_acc), "correct": int(task_correct), "total": int(limit), "source": src}
            total_accuracy += task_acc

            if generated_token_lens:
                per_task_generated_tokens.append(float(np.mean(generated_token_lens)))
            if answer_first:
                per_task_answer_first.append(float(np.mean(answer_first)))

        avg_accuracy = total_accuracy / len(task_names) if task_names else 0.0
        sec = {
            "generated_tokens_mean": float(np.mean(per_task_generated_tokens)) if per_task_generated_tokens else 0.0,
            "answer_first_rate": float(np.mean(per_task_answer_first)) if per_task_answer_first else 0.0,
        }
        return {"tasks": all_results, "average_accuracy": float(avg_accuracy), "tasks_evaluated": len(task_names), "secondary": sec}

    def _eval_efficiency(self, model, tokenizer, seed: int, gen_cfg: GenerationConfig) -> Dict[str, Any]:
        torch.manual_seed(seed)
        test_texts = [
            "Solve: 15 * 24 + 38  2",
            "Explain in one paragraph what a prime number is.",
        ]

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        total_time = 0.0
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(self.config.device)
            for _ in range(1):
                _ = model.generate(**inputs, max_new_tokens=16)
            start = time.time()
            _ = model.generate(
                **inputs,
                max_new_tokens=min(64, int(gen_cfg.max_new_tokens)),
                temperature=float(gen_cfg.temperature),
                do_sample=bool(gen_cfg.do_sample),
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
            )
            total_time += time.time() - start

        avg_inference_time = total_time / len(test_texts)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        else:
            memory_allocated = memory_reserved = 0.0

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "inference_speed_seconds": float(avg_inference_time),
            "memory_allocated_gb": float(memory_allocated),
            "memory_reserved_gb": float(memory_reserved),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        }
