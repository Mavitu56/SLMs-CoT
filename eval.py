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


def load_bbeh_task_dataset(task_name: str) -> Tuple[HFDataset, str]:
    """Loads a BBEH task JSON from GitHub with a small local cache."""

    base_cache = Path("/content") / "bbeh_cache"
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
        generation_cfg: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        generation_cfg = generation_cfg or self.config.eval_generation
        torch.manual_seed(seed)

        results: Dict[str, Any] = {
            "metadata": {
                "seed": seed,
                "use_cot_prompt": bool(use_cot_prompt),
                "generation": generation_cfg.to_jsonable(),
            }
        }

        if eval_gsm8k:
            results["gsm8k"] = self._eval_gsm8k(model, tokenizer, seed, use_cot_prompt, generation_cfg)
        if eval_bbh:
            results["bbh"] = self._eval_bbh(model, tokenizer, seed, generation_cfg)
        if eval_obqa:
            results["obqa"] = self._eval_obqa(model, tokenizer, seed, generation_cfg)
        if eval_efficiency:
            # Secondary metric: not part of hypothesis test by default.
            results["efficiency"] = self._eval_efficiency(model, tokenizer, seed, generation_cfg)

        # Scientific validity fix: keep a primary score based on task performance.
        results["primary_score"] = self._primary_score(results)
        # Backward-compatible aggregate (still recorded), but treated as secondary.
        results["overall_score"] = self._overall_score(results)
        return results

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

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_answer = generated[len(prompt) :].strip() if len(generated) > len(prompt) else generated.strip()
            pred = _extract_choice_letter(gen_answer, choices=valid_letters)
            if pred and gold and pred == gold:
                correct += 1
            total += 1

        return {"accuracy": correct / total if total else 0.0, "correct": int(correct), "total": int(total)}

    def _eval_gsm8k(self, model, tokenizer, seed: int, use_cot_prompt: bool, gen_cfg: GenerationConfig) -> Dict[str, Any]:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        limit = min(self.config.eval_limit_gsm8k, len(ds))
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), limit, replace=False)

        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        correct = 0
        total = 0
        for idx in tqdm(indices, desc="Eval GSM8K"):
            ex = ds[int(idx)]
            question = ex.get("question", "")
            gold = extract_gsm8k_answer(ex.get("answer", "") or ex.get("output", ""))

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

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = extract_gsm8k_answer(generated)
            if pred == gold and gold != "":
                correct += 1
            total += 1

        return {"accuracy": correct / total if total else 0.0, "correct": int(correct), "total": int(total)}

    def _eval_bbh(self, model, tokenizer, seed: int, gen_cfg: GenerationConfig) -> Dict[str, Any]:
        task_names = ["logical_deduction_five_objects", "causal_judgement", "formal_fallacies"]
        rng = np.random.RandomState(seed)

        ensure_tokenizer_has_pad(tokenizer, model)
        model.eval()

        all_results: Dict[str, Any] = {}
        total_accuracy = 0.0

        for task in task_names:
            ds, src = load_bbeh_task_dataset(task)
            train_split, eval_split = split_bbeh_dataset(ds, seed=seed, eval_fraction=0.2)
            ds = eval_split
            limit = min(self.config.eval_limit_bbh, len(ds))
            indices = rng.choice(len(ds), limit, replace=False)

            task_correct = 0
            for idx in tqdm(indices, desc=f"Eval BBH {task}"):
                ex = ds[int(idx)]
                input_text = ex.get("input", "") or ex.get("prompt", "")
                target = ex.get("target", "") or ex.get("output", "")

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
                gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
                gen_answer = gen[len(prompt):].strip() if len(gen) > len(prompt) else gen.strip()
                if bbh_answer_match(gen_answer, target or ""):
                    task_correct += 1

            task_acc = task_correct / limit if limit else 0.0
            all_results[task] = {"accuracy": float(task_acc), "correct": int(task_correct), "total": int(limit), "source": src}
            total_accuracy += task_acc

        avg_accuracy = total_accuracy / len(task_names) if task_names else 0.0
        return {"tasks": all_results, "average_accuracy": float(avg_accuracy), "tasks_evaluated": len(task_names)}

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
