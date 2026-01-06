from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import GenerationConfig, ensure_tokenizer_has_pad, get_safe_tokenizer_length, resolve_device, safe_model_to


def _dataset_fingerprint(dataset: Any, *, text_key: str = "text", max_examples: int = 128) -> str:
    """Best-effort dataset fingerprint for cache invalidation.

    Scientific traceability fix: caches must change when the dataset composition
    (including limits) changes, otherwise we risk silent reuse.
    """

    # HuggingFace Datasets often provide a stable fingerprint.
    fp = getattr(dataset, "_fingerprint", None)
    if isinstance(fp, str) and fp:
        return fp

    try:
        n = int(len(dataset))
    except Exception:
        n = -1

    h = hashlib.sha256()
    h.update(str(n).encode("utf-8"))

    # Sample a prefix of examples to avoid O(N) hashing.
    try:
        take = max_examples if n < 0 else min(n, max_examples)
        for i in range(int(take)):
            ex = dataset[i] if not isinstance(dataset, dict) else {text_key: dataset[text_key][i]}
            txt = ""
            if isinstance(ex, dict):
                txt = str(ex.get(text_key, ""))
            else:
                txt = str(ex)
            h.update(txt.encode("utf-8", errors="ignore"))
    except Exception:
        # If sampling fails, fall back to just length-based hash.
        pass

    return h.hexdigest()[:16]


def _dataset_domains(dataset: Any) -> Any:
    try:
        if hasattr(dataset, "unique"):
            return dataset.unique("domain")
    except Exception:
        pass
    return None


def _normalize_text_iterable(dataset: Any, *, text_key: str = "text") -> Any:
    """Normalize dataset-like input to an iterable of dicts containing text."""

    if isinstance(dataset, dict) and text_key in dataset and isinstance(dataset[text_key], list):
        return [{text_key: t} for t in dataset[text_key]]
    if isinstance(dataset, list):
        return [{text_key: t} for t in dataset]
    return dataset


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def make_cache_fingerprint(metadata: Dict[str, Any]) -> str:
    blob = _stable_json_dumps(metadata).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_cache_metadata(cache_path: Path, metadata: Dict[str, Any]) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "metadata.json").write_text(_stable_json_dumps(metadata), encoding="utf-8")


def read_cache_metadata(cache_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = cache_path / "metadata.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_versioned_cache_dir(root: Path, kind: str, metadata: Dict[str, Any]) -> Path:
    fingerprint = make_cache_fingerprint(metadata)
    return root / f"{kind}_{fingerprint}"


def cache_teacher_logits(
    teacher_model,
    tokenizer,
    dataset,
    cache_root: Path,
    batch_size: int,
    device: str,
    generation_cfg: Optional[GenerationConfig] = None,
    split: str = "train",
    seed: int = 42,
    kd_mode: str = "traditional",
    input_kind: str = "base",
    train_limit: Optional[int] = None,
    text_key: str = "text",
    max_length: Optional[int] = None,
) -> Path:
    """Compute teacher logits and persist shards.

    Note: logits caching depends on teacher identity, tokenizer max length, dataset split/seed,
    and generation_cfg (even if logits are from forward pass, config is tracked for traceability).
    """

    generation_cfg = generation_cfg or GenerationConfig(max_new_tokens=0, temperature=0.0, do_sample=False)

    dataset = _normalize_text_iterable(dataset, text_key=text_key)
    ds_fp = _dataset_fingerprint(dataset, text_key=text_key)
    effective_max_len = int(max_length or get_safe_tokenizer_length(tokenizer, fallback=512, upper_bound=4096))

    metadata = {
        "kind": "teacher_logits",
        "teacher": getattr(teacher_model, "name_or_path", None),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "max_length": effective_max_len,
        "batch_size": batch_size,
        "device": device,
        "split": split,
        "seed": seed,
        "kd_mode": kd_mode,
        "input_kind": input_kind,
        "train_limit": train_limit,
        "dataset_fingerprint": ds_fp,
        "dataset_domains": _dataset_domains(dataset),
        "generation": generation_cfg.to_jsonable(),
    }

    out_dir = resolve_versioned_cache_dir(cache_root, "teacher_logits", metadata)
    if (out_dir / "metadata.json").exists() and list(out_dir.glob("teacher_logits_shard_*.pt")):
        print(f" Reutilizando cache de logits: {out_dir}")
        return out_dir

    print(f" Gerando cache de logits: {out_dir}")
    write_cache_metadata(out_dir, metadata)

    target_device = resolve_device(device)
    teacher_model = safe_model_to(teacher_model, target_device)
    teacher_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    safe_max_len = effective_max_len

    for shard_idx, batch in enumerate(tqdm(loader, desc="Caching teacher logits")):
        prompts = [ex.get(text_key, "") for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            # Scientific validity fix: training uses padding='max_length'.
            padding="max_length",
            truncation=True,
            max_length=safe_max_len,
        )
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = teacher_model(
                output_hidden_states=False,
                output_attentions=False,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
            logits = out.logits.detach().cpu()

        payload = {
            "logits": logits.to(torch.float16),
            "input_ids": inputs["input_ids"].cpu(),
            "attention_mask": inputs["attention_mask"].cpu(),
        }
        shard_file = out_dir / f"teacher_logits_shard_{shard_idx}.pt"
        torch.save(payload, shard_file)

    return out_dir


def cache_teacher_cot(
    teacher_model,
    tokenizer,
    dataset,
    cache_root: Path,
    batch_size: int,
    device: str,
    generation_cfg: GenerationConfig,
    split: str = "train",
    seed: int = 42,
    prompt_max_length: Optional[int] = None,
    train_limit: Optional[int] = None,
    text_key: str = "text",
) -> Path:
    """Generate and persist Chain-of-Thought traces as JSONL."""

    dataset = _normalize_text_iterable(dataset, text_key=text_key)
    ds_fp = _dataset_fingerprint(dataset, text_key=text_key)
    metadata = {
        "kind": "teacher_cot",
        "teacher": getattr(teacher_model, "name_or_path", None),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "batch_size": batch_size,
        "device": device,
        "split": split,
        "seed": seed,
        "prompt_max_length": prompt_max_length,
        "train_limit": train_limit,
        "dataset_fingerprint": ds_fp,
        "dataset_domains": _dataset_domains(dataset),
        "generation": generation_cfg.to_jsonable(),
    }
    out_dir = resolve_versioned_cache_dir(cache_root, "teacher_cot", metadata)
    out_file = out_dir / "teacher_cot.jsonl"

    if out_file.exists() and (out_dir / "metadata.json").exists():
        print(f" Reutilizando cache de CoT: {out_file}")
        return out_file

    print(f" Gerando cache de CoT: {out_file}")
    write_cache_metadata(out_dir, metadata)

    target_device = resolve_device(device)
    teacher_model = safe_model_to(teacher_model, target_device)
    teacher_model.eval()

    desired_max = int(prompt_max_length or 2048)
    tokenizer_max = get_safe_tokenizer_length(tokenizer, fallback=desired_max, upper_bound=4096)
    max_len = min(desired_max, tokenizer_max)

    ensure_tokenizer_has_pad(tokenizer, teacher_model)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    with open(out_file, "w", encoding="utf-8") as fout:
        for batch in tqdm(loader, desc="Caching teacher CoT"):
            prompts = [
                f"Q: {ex.get('question', ex.get('text', ''))}\n"
                "A: Let's think step by step.\n### REASONING:\n"
                for ex in batch
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                # Keep stable sequence layout for reproducible truncation.
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            with torch.no_grad():
                generations = teacher_model.generate(
                    **inputs,
                    max_new_tokens=int(generation_cfg.max_new_tokens),
                    do_sample=bool(generation_cfg.do_sample),
                    temperature=float(generation_cfg.temperature),
                    top_p=generation_cfg.top_p,
                    top_k=generation_cfg.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=(generation_cfg.repetition_penalty or 1.0),
                )

            for example, output in zip(batch, generations):
                text = tokenizer.decode(output, skip_special_tokens=True)
                reasoning, answer = parse_cot_output(text)
                rec = {
                    "text": example.get("text", ""),
                    "question": example.get("question", example.get("text", "")),
                    "teacher_reasoning": reasoning,
                    "teacher_answer": answer,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_file


def parse_cot_output(generated_text: str):
    import re

    pattern_explicit = re.compile(
        r"(?:### REASONING:|\n\s*A:\s*Let's think step by step\.\s*### REASONING:)\s*(.*?)\s*(?:### FINAL_ANSWER:|\n\s*FINAL_ANSWER:)\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )
    pattern_implicit = re.compile(
        r"(.*?)\s*(?:Answer:|Final Answer:|Final:)\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    match_explicit = pattern_explicit.search(generated_text)
    if match_explicit:
        return match_explicit.group(1).strip(), match_explicit.group(2).strip()

    match_implicit = pattern_implicit.search(generated_text)
    if match_implicit:
        return match_implicit.group(1).strip(), match_implicit.group(2).strip()

    lines = generated_text.splitlines()
    if lines:
        answer = lines[-1].strip()
        reasoning = "\n".join(lines[:-1]).strip()
        if not reasoning:
            return generated_text.strip(), ""
        return reasoning, answer

    return "", ""
