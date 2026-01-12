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


def extract_gsm8k_answer(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    patterns = [
        r"answer[\s:\-]*([\$\d\.,]+)",
        r"final[\s:\-]*([\$\d\.,]+)",
        r"=[\s]*([\$\d\.,]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


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


def _resolve_bbeh_cache_dir() -> Path:
    """Resolve a persistent cache dir for BBEH downloads.

    Priority:
    1) SLM_BBEH_CACHE_DIR (explicit override)
    2) SLM_DRIVE_ROOT/scientific_results/cache/bbeh_cache
    3) Colab Drive default (when mounted)
    4) /content/bbeh_cache (Colab ephemeral)
    5) ./bbeh_cache (local fallback)
    """

    env = os.environ.get("SLM_BBEH_CACHE_DIR")
    if env:
        return Path(env)

    drive_root = os.environ.get("SLM_DRIVE_ROOT")
    if drive_root:
        return Path(drive_root) / "scientific_results" / "cache" / "bbeh_cache"

    try:
        if os.path.ismount("/content/drive") and Path("/content/drive/MyDrive").is_dir():
            return Path("/content/drive/MyDrive/SLM_results") / "scientific_results" / "cache" / "bbeh_cache"
    except Exception:
        pass

    if Path("/content").exists():
        return Path("/content") / "bbeh_cache"
    return Path.cwd() / "bbeh_cache"


def load_bbeh_task_dataset(task_name: str) -> Tuple[HFDataset, str]:
    """Loads a BBEH task JSON from GitHub with a small local cache."""

    base_cache = _resolve_bbeh_cache_dir()
    base_cache.mkdir(parents=True, exist_ok=True)

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
            dataset = _load_bbeh_task_from_repo(base_cache, directory)
            if dataset is not None:
                return dataset, directory
        except Exception as err:
            last_err = err
            continue
    raise RuntimeError(f"No foi possvel carregar task {task_name}: {last_err}")


def _load_bbeh_task_from_repo(cache_dir: Path, directory_name: str) -> HFDataset:
    from datasets import Dataset as HFDataset

    base_url = "https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks"
    cache_file = cache_dir / f"{directory_name}.json"

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        url = f"{base_url}/{directory_name}/task.json"
        print(f" Baixando task '{directory_name}' do BBEH...")
        response = requests.get(url, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"No foi possvel baixar {directory_name} ({response.status_code}).")
        payload = response.json()
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

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

        for idx in tqdm(indices, desc="Eval OBQA"):
            ex = ds[int(idx)]
            stem = str(ex.get("question_stem", ""))
            choices_obj = ex.get("choices") or {}
            texts = list(choices_obj.get("text") or [])
            labels = list(choices_obj.get("label") or [])
            gold = str(ex.get("answerKey", "")).strip().upper()

            # Build map label->text and keep stable order A,B,C,D when possible.
            pairs = []
            for lab, txt in zip(labels, texts):
                lab_u = str(lab).strip().upper()
                if lab_u and txt is not None:
                    pairs.append((lab_u, str(txt)))
            if not pairs:
                continue

            # Stable ordering by label.
            pairs = sorted(pairs, key=lambda p: p[0])
            valid_letters = "".join([p[0] for p in pairs])

            lines = [f"Q: {stem}", "Choices:"]
            for lab, txt in pairs:
                lines.append(f"{lab}) {txt}")
            prompt = "\n".join(lines) + "\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(8, int(gen_cfg.max_new_tokens)),
                    temperature=float(gen_cfg.temperature),
                    do_sample=bool(gen_cfg.do_sample),
                    top_p=gen_cfg.top_p,
                    top_k=gen_cfg.top_k,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                )

            try:
                input_len = int(inputs["input_ids"].shape[1])
                out_len = int(outputs.shape[1])
                generated_token_lens.append(max(0, out_len - input_len))
            except Exception:
                pass

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_answer = generated[len(prompt) :].strip() if len(generated) > len(prompt) else generated.strip()
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
        for idx in tqdm(indices, desc="Eval GSM8K"):
            ex = ds[int(idx)]
            question = ex.get("question", "")
            gold = extract_gsm8k_answer(ex.get("answer", "") or ex.get("output", ""))

            if bool(cascod_two_stage):
                prompt_r = build_cascod_rationale_prompt(question, granularity_level=0)
                inputs_r = tokenizer(prompt_r, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                with torch.inference_mode():
                    out_r = model.generate(
                        **inputs_r,
                        max_new_tokens=int(gen_cfg.max_new_tokens),
                        temperature=float(gen_cfg.temperature),
                        do_sample=bool(gen_cfg.do_sample),
                        top_p=gen_cfg.top_p,
                        top_k=gen_cfg.top_k,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                    )

                rationale = ""
                r_cont_len = 0
                try:
                    in_len = int(inputs_r["input_ids"].shape[1])
                    out_len = int(out_r.shape[1])
                    r_cont_len = max(0, out_len - in_len)
                except Exception:
                    r_cont_len = 0

                r_full = tokenizer.decode(out_r[0], skip_special_tokens=True)
                rationale = r_full[len(prompt_r) :].strip() if len(r_full) > len(prompt_r) else r_full.strip()

                prompt_a = build_cascod_answer_prompt(question, rationale)
                inputs_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                with torch.inference_mode():
                    out_a = model.generate(
                        **inputs_a,
                        max_new_tokens=min(32, int(gen_cfg.max_new_tokens)),
                        temperature=float(gen_cfg.temperature),
                        do_sample=bool(gen_cfg.do_sample),
                        top_p=gen_cfg.top_p,
                        top_k=gen_cfg.top_k,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                    )

                a_cont_len = 0
                try:
                    in_len = int(inputs_a["input_ids"].shape[1])
                    out_len = int(out_a.shape[1])
                    a_cont_len = max(0, out_len - in_len)
                except Exception:
                    a_cont_len = 0

                generated_token_lens.append(int(r_cont_len) + int(a_cont_len))

                a_full = tokenizer.decode(out_a[0], skip_special_tokens=True)
                cont_text = a_full[len(prompt_a) :].strip() if len(a_full) > len(prompt_a) else a_full.strip()
                pred = extract_gsm8k_answer(cont_text)

                # Answer-first: numeric-looking continuation for stage 2.
                answer_first.append(1 if _starts_with_answer_like(cont_text, mode="numeric") else 0)

                # CoT token proxy: total continuation minus tokens for extracted final answer.
                if (int(r_cont_len) + int(a_cont_len)) > 0:
                    try:
                        ans_ids = tokenizer(pred or "", add_special_tokens=False).get("input_ids") or []
                        ans_len = int(len(ans_ids))
                    except Exception:
                        ans_len = 0
                    cot_token_lens.append(max(0, int(r_cont_len) + int(a_cont_len) - int(ans_len)))

                # Simple rationale/answer consistency: compare last number in rationale vs extracted answer.
                r_last = _normalize_number_string(_extract_last_number_like(rationale))
                a_norm = _normalize_number_string(pred)
                if r_last and a_norm:
                    consistency.append(1 if r_last == a_norm else 0)

            else:
                if use_cot_prompt:
                    prompt = f"Q: {question}\nA: Let's think step by step."
                else:
                    prompt = f"Q: {question}\nA:"

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=int(gen_cfg.max_new_tokens),
                        temperature=float(gen_cfg.temperature),
                        do_sample=bool(gen_cfg.do_sample),
                        top_p=gen_cfg.top_p,
                        top_k=gen_cfg.top_k,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                    )

                cont_text = ""
                try:
                    input_len = int(inputs["input_ids"].shape[1])
                    out_len = int(outputs.shape[1])
                    cont_len = max(0, out_len - input_len)
                    generated_token_lens.append(cont_len)
                except Exception:
                    cont_len = 0

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                cont_text = generated[len(prompt) :].strip() if len(generated) > len(prompt) else generated.strip()
                if (not use_cot_prompt) and bool(answer_first_eval):
                    pred = extract_gsm8k_answer_first(cont_text)
                else:
                    pred = extract_gsm8k_answer(generated)
            if pred == gold and gold != "":
                correct += 1
            total += 1

            if not bool(cascod_two_stage):
                # Answer-first: numeric-looking continuation.
                answer_first.append(1 if _starts_with_answer_like(cont_text, mode="numeric") else 0)

                # CoT token proxy: generated tokens minus tokens for extracted final answer.
                if use_cot_prompt and cont_len > 0:
                    try:
                        ans_ids = tokenizer(pred or "", add_special_tokens=False).get("input_ids") or []
                        ans_len = int(len(ans_ids))
                    except Exception:
                        ans_len = 0
                    cot_token_lens.append(max(0, int(cont_len) - int(ans_len)))

                # Simple rationale/answer consistency: compare last number in 'reasoning' vs final extracted answer.
                if use_cot_prompt:
                    # Prefer '####' split when present.
                    if "####" in generated:
                        reasoning_text = generated.split("####")[0]
                    else:
                        # Fallback: look before the last occurrence of the extracted answer string.
                        if pred:
                            pos = generated.rfind(str(pred))
                            reasoning_text = generated[:pos] if pos > 0 else generated
                        else:
                            reasoning_text = generated
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
            for idx in tqdm(indices, desc=f"Eval BBH {task}"):
                ex = ds[int(idx)]
                input_text = ex.get("input", "") or ex.get("prompt", "")
                target = ex.get("target", "") or ex.get("output", "")

                if bool(cascod_two_stage):
                    prompt_r = build_cascod_rationale_prompt(input_text, granularity_level=0)
                    inputs_r = tokenizer(prompt_r, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                    with torch.inference_mode():
                        out_r = model.generate(
                            **inputs_r,
                            max_new_tokens=int(gen_cfg.max_new_tokens),
                            temperature=float(gen_cfg.temperature),
                            do_sample=bool(gen_cfg.do_sample),
                            top_p=gen_cfg.top_p,
                            top_k=gen_cfg.top_k,
                            pad_token_id=getattr(tokenizer, "eos_token_id", None),
                            repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                        )

                    rationale = ""
                    r_cont_len = 0
                    try:
                        in_len = int(inputs_r["input_ids"].shape[1])
                        out_len = int(out_r.shape[1])
                        r_cont_len = max(0, out_len - in_len)
                    except Exception:
                        r_cont_len = 0

                    r_full = tokenizer.decode(out_r[0], skip_special_tokens=True)
                    rationale = r_full[len(prompt_r) :].strip() if len(r_full) > len(prompt_r) else r_full.strip()

                    prompt_a = build_cascod_answer_prompt(input_text, rationale)
                    inputs_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                    with torch.inference_mode():
                        out_a = model.generate(
                            **inputs_a,
                            max_new_tokens=min(64, int(gen_cfg.max_new_tokens)),
                            temperature=float(gen_cfg.temperature),
                            do_sample=bool(gen_cfg.do_sample),
                            top_p=gen_cfg.top_p,
                            top_k=gen_cfg.top_k,
                            pad_token_id=getattr(tokenizer, "eos_token_id", None),
                            repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                        )

                    a_cont_len = 0
                    try:
                        in_len = int(inputs_a["input_ids"].shape[1])
                        out_len = int(out_a.shape[1])
                        a_cont_len = max(0, out_len - in_len)
                    except Exception:
                        a_cont_len = 0

                    generated_token_lens.append(int(r_cont_len) + int(a_cont_len))

                    a_full = tokenizer.decode(out_a[0], skip_special_tokens=True)
                    gen_answer = a_full[len(prompt_a):].strip() if len(a_full) > len(prompt_a) else a_full.strip()

                else:
                    prompt = f"{input_text}\nAnswer:"
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.config.device)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(64, int(gen_cfg.max_new_tokens)),
                            temperature=float(gen_cfg.temperature),
                            do_sample=bool(gen_cfg.do_sample),
                            top_p=gen_cfg.top_p,
                            top_k=gen_cfg.top_k,
                            pad_token_id=getattr(tokenizer, "eos_token_id", None),
                            repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                        )
                    try:
                        input_len = int(inputs["input_ids"].shape[1])
                        out_len = int(outputs.shape[1])
                        generated_token_lens.append(max(0, out_len - input_len))
                    except Exception:
                        pass
                    gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    gen_answer = gen[len(prompt):].strip() if len(gen) > len(prompt) else gen.strip()

                answer_first.append(1 if _starts_with_answer_like(gen_answer, mode="bool") or _starts_with_answer_like(gen_answer, mode="letter") else 0)
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
