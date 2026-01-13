from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import GenerationConfig, ensure_tokenizer_has_pad, get_safe_tokenizer_length, resolve_device, safe_model_to
from prompts import (
    PROMPT_VERSION,
    build_cot_prompt,
    build_one_shot_demo,
    build_one_shot_teacher_demo_for_post_cot_gold_rationale,
    build_teacher_cot_prompt,
)


def _normalize_answer(ans: str) -> str:
    a = (ans or "").strip()
    # Conservative normalization: avoid over-normalizing (BBH tasks can be free-form).
    a = a.replace("\n", " ").strip()
    return a.lower()


def _extract_gold_answer(example: Dict[str, Any]) -> str:
    # Prefer already-extracted final answers (GSM8K pipeline provides this).
    for key in ("final_answer", "answer", "target", "label", "output"):
        v = example.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Some datasets store numeric answers.
    v = example.get("final_answer")
    if v is not None:
        return str(v).strip()
    return ""


def tokenizer_fingerprint(tokenizer) -> str:
    """Public tokenizer fingerprint helper.

    Used across the repo to ensure logits-KD comparisons are valid and caches
    don't silently collide across tokenizer variants.
    """

    return _tokenizer_fingerprint(tokenizer)


def _tokenizer_fingerprint(tokenizer) -> str:
    """Stable tokenizer identity hash.

    Scientific validity fix: logits-KD requires token IDs to match between
    teacher and student. For families like Qwen/Gemma/Llama, different model
    sizes often share an identical tokenizer even though `name_or_path` differs.
    """

    cached = getattr(tokenizer, "_slm_tokenizer_hash", None)
    if isinstance(cached, str) and cached:
        return cached

    vocab = tokenizer.get_vocab()  # token -> id
    h = hashlib.sha256()
    h.update(tokenizer.__class__.__name__.encode("utf-8"))
    h.update(str(len(vocab)).encode("utf-8"))
    for k in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        h.update(f"|{k}={getattr(tokenizer, k, None)}".encode("utf-8"))

    # Full mapping hash (deterministic order).
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[0]):
        h.update(token.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(str(int(idx)).encode("utf-8"))
        h.update(b"\n")

    digest = h.hexdigest()[:16]
    try:
        setattr(tokenizer, "_slm_tokenizer_hash", digest)
    except Exception:
        pass
    return digest


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
        "tokenizer_hash": _tokenizer_fingerprint(tokenizer),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": int(len(tokenizer.get_vocab())),
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

    def _env_flag(name: str, default: str = "0") -> bool:
        v = os.environ.get(name, default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None:
            return int(default)
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    def _env_float(name: str, default: float) -> float:
        v = os.environ.get(name)
        if v is None:
            return float(default)
        try:
            return float(str(v).strip())
        except Exception:
            return float(default)

    cache_debug = _env_flag("SLM_CACHE_DEBUG", "1")
    cache_log_every = max(1, _env_int("SLM_CACHE_LOG_EVERY", 50))
    sanitize_logits = _env_flag("SLM_CACHE_SANITIZE_LOGITS", "1")
    clamp_for_fp16 = _env_flag("SLM_CACHE_CLAMP_FP16", "1")
    fp16_safe_abs = _env_float("SLM_CACHE_FP16_SAFE_ABS", 60000.0)
    shard_stats_path = out_dir / "shard_stats.jsonl"

    target_device = resolve_device(device)
    teacher_model = safe_model_to(teacher_model, target_device)
    teacher_model.eval()

    # Some LMs ship without a pad token; caching uses padding='max_length'.
    ensure_tokenizer_has_pad(tokenizer, teacher_model)

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
            logits = out.logits.detach().cpu().to(torch.float32)

        # Diagnostics + optional sanitization before fp16 cast.
        nonfinite = (~torch.isfinite(logits)).sum().item()
        max_abs = float(logits.abs().max().item()) if logits.numel() else 0.0
        above_fp16 = int((logits.abs() > fp16_safe_abs).sum().item()) if logits.numel() else 0

        if cache_debug and (nonfinite or above_fp16 or ((shard_idx + 1) % cache_log_every == 0)):
            print(
                f" (cache) shard={shard_idx} max|logit|={max_abs:.2f} nonfinite={int(nonfinite)} above_fp16={int(above_fp16)}"
            )

        logits_to_save = logits
        if sanitize_logits:
            logits_to_save = torch.nan_to_num(logits_to_save, nan=0.0, posinf=fp16_safe_abs, neginf=-fp16_safe_abs)
        if clamp_for_fp16 and fp16_safe_abs > 0:
            logits_to_save = logits_to_save.clamp(min=-fp16_safe_abs, max=fp16_safe_abs)

        logits_fp16 = logits_to_save.to(torch.float16)
        nonfinite_fp16 = (~torch.isfinite(logits_fp16)).sum().item()

        try:
            # Persist per-shard stats for postmortem scanning.
            rec = {
                "shard": int(shard_idx),
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "max_abs": float(max_abs),
                "nonfinite": int(nonfinite),
                "above_fp16": int(above_fp16),
                "nonfinite_after_fp16": int(nonfinite_fp16),
            }
            with open(shard_stats_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        payload = {
            "logits": logits_fp16,
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
    granularity_level: int = 0,
    granularity_multi_level: bool = False,
    granularity_levels: Optional[List[int]] = None,
    post_cot: bool = False,
    one_shot: bool = False,
    post_cot_gold_rationale: bool = False,
    post_cot_use_ig: bool = False,
    post_cot_ig_steps: int = 8,
    post_cot_ig_top_frac: float = 0.3,
    filter_by_gold_answer: bool = False,
) -> Path:
    """Generate and persist Chain-of-Thought traces as JSONL."""

    dataset = _normalize_text_iterable(dataset, text_key=text_key)
    ds_fp = _dataset_fingerprint(dataset, text_key=text_key)
    metadata = {
        "kind": "teacher_cot",
        "teacher": getattr(teacher_model, "name_or_path", None),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "tokenizer_hash": _tokenizer_fingerprint(tokenizer),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": int(len(tokenizer.get_vocab())),
        "batch_size": batch_size,
        "device": device,
        "split": split,
        "seed": seed,
        "prompt_max_length": prompt_max_length,
        "prompt_version": PROMPT_VERSION,
        "granularity_level": int(granularity_level or 0),
        "granularity_multi_level": bool(granularity_multi_level),
        "granularity_levels": list(granularity_levels) if isinstance(granularity_levels, list) else None,
        "post_cot": bool(post_cot),
        "one_shot": bool(one_shot),
        "post_cot_gold_rationale": bool(post_cot_gold_rationale),
        "post_cot_use_ig": bool(post_cot_use_ig),
        "post_cot_ig_steps": int(post_cot_ig_steps or 0),
        "post_cot_ig_top_frac": float(post_cot_ig_top_frac or 0.0),
        "filter_by_gold_answer": bool(filter_by_gold_answer),
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

    # Resolve granularity levels to cache (multi-level pipeline).
    resolved_levels: List[int] = []
    if bool(granularity_multi_level) and int(granularity_level or 0) > 0:
        if isinstance(granularity_levels, list) and granularity_levels:
            for x in granularity_levels:
                try:
                    v = int(x)
                except Exception:
                    continue
                if 1 <= v <= 6:
                    resolved_levels.append(v)
        if not resolved_levels:
            resolved_levels = list(range(1, max(1, min(6, int(granularity_level or 0))) + 1))
        resolved_levels = sorted(set(resolved_levels))

    use_ig = bool(post_cot_use_ig) and bool(post_cot) and bool(post_cot_gold_rationale)
    if use_ig:
        try:
            import captum  # noqa: F401
        except Exception:
            raise RuntimeError(
                "post_cot_use_ig=True requer o pacote 'captum'. Instale com: pip install captum"
            )

    def _ig_filter_reasoning(*, question: str, reasoning: str, answer: str) -> Tuple[Optional[str], Dict[str, Any]]:
        meta: Dict[str, Any] = {"enabled": True, "steps": int(post_cot_ig_steps or 0), "top_frac": float(post_cot_ig_top_frac or 0.0)}
        q = (question or "").strip()
        r = (reasoning or "").strip()
        a = (answer or "").strip()
        if not q or not r or not a:
            meta["skipped"] = True
            meta["reason"] = "missing_q_r_a"
            return None, meta

        try:
            # Attribution prompt: rationale -> answer (importance of rationale tokens for the answer).
            prompt = f"Q: {q}\n### REASONING:\n{r}\n### FINAL_ANSWER:\n"
            ans_ids = tokenizer(" " + a, add_special_tokens=False).get("input_ids") or []
            if not ans_ids:
                ans_ids = tokenizer(a, add_special_tokens=False).get("input_ids") or []
            if not ans_ids:
                meta["skipped"] = True
                meta["reason"] = "no_answer_tokens"
                return None, meta
            target_id = int(ans_ids[0])

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"].to(target_device)
            attention_mask = enc["attention_mask"].to(target_device)
            offsets = enc.get("offset_mapping")
            if offsets is None:
                meta["skipped"] = True
                meta["reason"] = "no_offset_mapping"
                return None, meta

            # Baseline: pad token IDs.
            baseline_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
            baselines = torch.full_like(input_ids, baseline_id)

            def forward_func(ids, mask):
                out = teacher_model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits = out.logits
                return logits[:, -1, target_id]

            from captum.attr import LayerIntegratedGradients

            lig = LayerIntegratedGradients(forward_func, teacher_model.get_input_embeddings())

            with torch.enable_grad():
                attributions = lig.attribute(
                    inputs=input_ids,
                    baselines=baselines,
                    additional_forward_args=(attention_mask,),
                    n_steps=int(post_cot_ig_steps or 8),
                )

            # Token-level importance score.
            scores = attributions.detach().abs().sum(dim=-1)[0].to("cpu")  # (L,)
            offsets = offsets[0].to("cpu").tolist()

            span_start = prompt.find(r)
            if span_start < 0:
                meta["skipped"] = True
                meta["reason"] = "reasoning_span_not_found"
                return None, meta
            span_end = span_start + len(r)

            reasoning_token_idxs: List[int] = []
            reasoning_spans: List[Tuple[int, int]] = []
            for i, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                if s >= span_start and e <= span_end:
                    reasoning_token_idxs.append(i)
                    reasoning_spans.append((int(s), int(e)))

            if not reasoning_token_idxs:
                meta["skipped"] = True
                meta["reason"] = "no_reasoning_tokens"
                return None, meta

            k = max(1, int(round(len(reasoning_token_idxs) * float(post_cot_ig_top_frac or 0.0))))
            if k <= 0:
                meta["skipped"] = True
                meta["reason"] = "top_frac_zero"
                return None, meta

            # Pick top-k tokens in the reasoning span.
            scored = []
            for idx, (s, e) in zip(reasoning_token_idxs, reasoning_spans):
                scored.append((float(scores[int(idx)].item()), int(s), int(e)))
            scored.sort(key=lambda x: x[0], reverse=True)
            keep = scored[:k]
            keep_spans = sorted([(s, e) for _, s, e in keep], key=lambda x: (x[0], x[1]))

            # Merge overlapping/adjacent spans.
            merged: List[List[int]] = []
            for s, e in keep_spans:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(int(merged[-1][1]), int(e))

            pieces = []
            for s, e in merged:
                chunk = prompt[int(s) : int(e)].strip()
                if chunk:
                    pieces.append(chunk)
            filtered = " ".join(pieces).strip()

            meta.update({"k": int(k), "n_reasoning_tokens": int(len(reasoning_token_idxs)), "n_kept_spans": int(len(merged))})
            return filtered if filtered else None, meta
        except Exception as exc:
            meta["error"] = str(exc)
            return None, meta

    # Optional 1-shot exemplar (deterministic).
    one_shot_teacher_prefix: Optional[str] = None
    one_shot_student_prefix: Optional[str] = None
    if bool(one_shot):
        try:
            n = int(len(dataset))
        except Exception:
            n = 0

        exemplar = None
        if n > 0 and hasattr(dataset, "__getitem__"):
            try:
                exemplar = dataset[int(seed) % max(1, n)]
            except Exception:
                exemplar = None
        if exemplar is None:
            try:
                exemplar = next(iter(dataset))
            except Exception:
                exemplar = None

        if isinstance(exemplar, dict):
            ex_q = exemplar.get("question", exemplar.get(text_key, ""))
            ex_gold = _extract_gold_answer(exemplar)
            ex_teacher_prompt, _ = build_teacher_cot_prompt(
                str(ex_q),
                granularity_level=int(granularity_level or 0),
                post_cot=bool(post_cot),
                one_shot_prefix=None,
                gold_answer=ex_gold,
                post_cot_gold_rationale=bool(post_cot_gold_rationale),
            )
            ex_inputs = tokenizer(
                [ex_teacher_prompt],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            ex_inputs = {k: v.to(target_device) for k, v in ex_inputs.items()}
            with torch.no_grad():
                ex_gen = teacher_model.generate(
                    **ex_inputs,
                    max_new_tokens=int(generation_cfg.max_new_tokens),
                    do_sample=bool(generation_cfg.do_sample),
                    temperature=float(generation_cfg.temperature),
                    top_p=generation_cfg.top_p,
                    top_k=generation_cfg.top_k,
                    pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
                    repetition_penalty=(generation_cfg.repetition_penalty or 1.0),
                )

            ex_in_len = int(ex_inputs["attention_mask"].sum(dim=1)[0].item())
            ex_cont = ex_gen[0][ex_in_len:]
            ex_cont_text = tokenizer.decode(ex_cont, skip_special_tokens=True).strip()

            if bool(post_cot) and bool(post_cot_gold_rationale):
                ex_reasoning = ex_cont_text
                ex_answer = ex_gold
            else:
                ex_full_text = ex_teacher_prompt + ex_cont_text
                ex_reasoning, ex_answer = parse_cot_output(ex_full_text)

            ex_reasoning = (ex_reasoning or "").strip()
            ex_answer = (ex_answer or ex_gold or "").strip()

            # Save a compact exemplar identity in metadata for traceability.
            metadata["one_shot_exemplar"] = {
                "question": str(ex_q)[:200],
                "gold_answer": str(ex_gold)[:80],
            }
            # Prefix shown to student.
            one_shot_student_prefix = build_one_shot_demo(
                question=str(ex_q),
                answer=str(ex_answer),
                reasoning=str(ex_reasoning),
                post_cot=bool(post_cot),
            )
            # Prefix shown to teacher.
            if bool(post_cot) and bool(post_cot_gold_rationale):
                one_shot_teacher_prefix = build_one_shot_teacher_demo_for_post_cot_gold_rationale(
                    question=str(ex_q),
                    answer=str(ex_answer),
                    reasoning=str(ex_reasoning),
                )
            else:
                one_shot_teacher_prefix = one_shot_student_prefix

    # Re-write metadata if exemplar was added.
    write_cache_metadata(out_dir, metadata)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    with open(out_file, "w", encoding="utf-8") as fout:
        n_total = 0
        n_kept = 0
        n_filtered = 0
        for batch in tqdm(loader, desc="Caching teacher CoT"):
            # Multi-level: cache multiple granularity levels per example.
            if resolved_levels:
                per_ex = []
                for ex in batch:
                    q = ex.get("question", ex.get(text_key, ""))
                    gold = _extract_gold_answer(ex)
                    per_ex.append(
                        {
                            "example": ex,
                            "question": str(q),
                            "gold": str(gold),
                            "prompt_levels": {},
                            "teacher_prompt_levels": {},
                            "teacher_reasoning_levels": {},
                            "teacher_answer_levels": {},
                            "teacher_reasoning_token_lens": {},
                        }
                    )

                for lvl in resolved_levels:
                    lvl_prompts = []
                    lvl_student_prompts = []
                    for exd in per_ex:
                        teacher_prompt, _ = build_teacher_cot_prompt(
                            exd["question"],
                            granularity_level=int(lvl),
                            post_cot=bool(post_cot),
                            one_shot_prefix=one_shot_teacher_prefix,
                            gold_answer=exd["gold"],
                            post_cot_gold_rationale=bool(post_cot_gold_rationale),
                        )
                        student_prompt, _ = build_cot_prompt(
                            exd["question"],
                            granularity_level=int(lvl),
                            post_cot=bool(post_cot),
                            one_shot_prefix=one_shot_student_prefix,
                        )
                        lvl_prompts.append(teacher_prompt)
                        lvl_student_prompts.append(student_prompt)

                    inputs = tokenizer(
                        lvl_prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_len,
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                    with torch.no_grad():
                        generations = teacher_model.generate(
                            **inputs,
                            max_new_tokens=int(generation_cfg.max_new_tokens),
                            do_sample=bool(generation_cfg.do_sample),
                            temperature=float(generation_cfg.temperature),
                            top_p=generation_cfg.top_p,
                            top_k=generation_cfg.top_k,
                            pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
                            repetition_penalty=(generation_cfg.repetition_penalty or 1.0),
                        )

                    for exd, teacher_prompt, student_prompt, out_ids, in_len in zip(
                        per_ex,
                        lvl_prompts,
                        lvl_student_prompts,
                        generations,
                        input_lens,
                    ):
                        cont_ids = out_ids[int(in_len) :]
                        cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

                        if bool(post_cot) and bool(post_cot_gold_rationale):
                            answer = (exd["gold"] or "").strip()
                            reasoning = cont_text
                            if reasoning.lower().startswith("### reasoning:"):
                                reasoning = reasoning[len("### reasoning:") :].strip()
                        else:
                            generated_text = str(teacher_prompt) + ("\n" + cont_text if cont_text else "")
                            reasoning, answer = parse_cot_output(generated_text)

                        exd["prompt_levels"][str(int(lvl))] = student_prompt
                        exd["teacher_prompt_levels"][str(int(lvl))] = teacher_prompt
                        exd["teacher_reasoning_levels"][str(int(lvl))] = (reasoning or "").strip()
                        exd["teacher_answer_levels"][str(int(lvl))] = (answer or "").strip()
                        try:
                            rid = tokenizer((reasoning or ""), add_special_tokens=False).get("input_ids") or []
                            exd["teacher_reasoning_token_lens"][str(int(lvl))] = int(len(rid))
                        except Exception:
                            exd["teacher_reasoning_token_lens"][str(int(lvl))] = 0

                base_lvl = str(int(max(resolved_levels) if resolved_levels else int(granularity_level or 0)))
                for exd in per_ex:
                    n_total += 1
                    gold = exd["gold"]
                    reasoning = (exd["teacher_reasoning_levels"].get(base_lvl) or "").strip()
                    answer = (exd["teacher_answer_levels"].get(base_lvl) or "").strip()

                    gold_norm = _normalize_answer(str(gold))
                    ans_norm = _normalize_answer(str(answer))
                    if bool(filter_by_gold_answer) and gold_norm and ans_norm and (gold_norm != ans_norm):
                        n_filtered += 1
                        continue

                    n_kept += 1

                    rec = {
                        "text": exd["example"].get("text", ""),
                        "question": exd["example"].get("question", exd["example"].get("text", "")),
                        "gold_answer": gold,
                        # Backward-compatible single prompt (default to base level)
                        "prompt": exd["prompt_levels"].get(base_lvl, ""),
                        "teacher_prompt": exd["teacher_prompt_levels"].get(base_lvl, ""),
                        "teacher_reasoning": reasoning,
                        "teacher_answer": answer,
                        # Multi-level fields
                        "granularity_levels": list(resolved_levels),
                        "prompt_levels": exd["prompt_levels"],
                        "teacher_prompt_levels": exd["teacher_prompt_levels"],
                        "teacher_reasoning_levels": exd["teacher_reasoning_levels"],
                        "teacher_answer_levels": exd["teacher_answer_levels"],
                        "teacher_reasoning_token_lens": exd["teacher_reasoning_token_lens"],
                    }
                    # Alignment diagnostic: are token lengths monotonic with level?
                    try:
                        lens = [int(exd["teacher_reasoning_token_lens"].get(str(l), 0)) for l in resolved_levels]
                        rec["granularity_monotonic_by_len"] = bool(all(lens[i] <= lens[i + 1] for i in range(len(lens) - 1)))
                    except Exception:
                        rec["granularity_monotonic_by_len"] = None

                    # Optional Post-CoT IG filtering (store filtered rationale but keep original).
                    if use_ig:
                        filtered, ig_meta = _ig_filter_reasoning(question=exd["question"], reasoning=reasoning, answer=(gold or answer))
                        rec["teacher_reasoning_ig"] = (filtered or "")
                        rec["teacher_reasoning_ig_meta"] = ig_meta

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            else:
                # Single-level (legacy)
                prompts = []
                student_prompts = []
                gold_answers = []
                for ex in batch:
                    q = ex.get("question", ex.get(text_key, ""))
                    gold = _extract_gold_answer(ex)
                    gold_answers.append(gold)

                    teacher_prompt, _ = build_teacher_cot_prompt(
                        str(q),
                        granularity_level=int(granularity_level or 0),
                        post_cot=bool(post_cot),
                        one_shot_prefix=one_shot_teacher_prefix,
                        gold_answer=gold,
                        post_cot_gold_rationale=bool(post_cot_gold_rationale),
                    )
                    student_prompt, _ = build_cot_prompt(
                        str(q),
                        granularity_level=int(granularity_level or 0),
                        post_cot=bool(post_cot),
                        one_shot_prefix=one_shot_student_prefix,
                    )
                    prompts.append(teacher_prompt)
                    student_prompts.append(student_prompt)
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    # Keep stable sequence layout for reproducible truncation.
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                with torch.no_grad():
                    generations = teacher_model.generate(
                        **inputs,
                        max_new_tokens=int(generation_cfg.max_new_tokens),
                        do_sample=bool(generation_cfg.do_sample),
                        temperature=float(generation_cfg.temperature),
                        top_p=generation_cfg.top_p,
                        top_k=generation_cfg.top_k,
                        pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
                        repetition_penalty=(generation_cfg.repetition_penalty or 1.0),
                    )

                for example, teacher_prompt, student_prompt, gold, out_ids, in_len in zip(
                    batch,
                    prompts,
                    student_prompts,
                    gold_answers,
                    generations,
                    input_lens,
                ):
                    n_total += 1
                    cont_ids = out_ids[int(in_len) :]
                    cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

                    if bool(post_cot) and bool(post_cot_gold_rationale):
                        answer = (gold or "").strip()
                        reasoning = cont_text
                        # If the model repeats the marker, drop it.
                        if reasoning.lower().startswith("### reasoning:"):
                            reasoning = reasoning[len("### reasoning:") :].strip()
                    else:
                        generated_text = str(teacher_prompt) + ("\n" + cont_text if cont_text else "")
                        reasoning, answer = parse_cot_output(generated_text)

                    gold_norm = _normalize_answer(str(gold))
                    ans_norm = _normalize_answer(str(answer))
                    if bool(filter_by_gold_answer) and gold_norm and ans_norm and (gold_norm != ans_norm):
                        n_filtered += 1
                        continue

                    n_kept += 1
                    rec = {
                        "text": example.get("text", ""),
                        "question": example.get("question", example.get("text", "")),
                        "gold_answer": gold,
                        # Student-side prompt (used in distillation/logits caches).
                        "prompt": student_prompt,
                        # Teacher-side prompt (for traceability).
                        "teacher_prompt": teacher_prompt,
                        "teacher_reasoning": (reasoning or "").strip(),
                        "teacher_answer": (answer or "").strip(),
                    }
                    if use_ig:
                        ex_q = str(example.get("question", example.get(text_key, "")) or "")
                        filtered, ig_meta = _ig_filter_reasoning(question=ex_q, reasoning=(reasoning or ""), answer=str(gold or answer))
                        rec["teacher_reasoning_ig"] = (filtered or "")
                        rec["teacher_reasoning_ig_meta"] = ig_meta
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if bool(filter_by_gold_answer):
            print(f" CasCoD filter_by_gold_answer: kept={n_kept}/{n_total}, filtered={n_filtered}")

    return out_file


def parse_cot_output(generated_text: str):
    import re

    # Post-CoT (answer first):
    #   ### FINAL_ANSWER: <answer>\n### REASONING: <reasoning>
    pattern_post = re.compile(
        r"###\s*FINAL_ANSWER\s*:\s*(.*?)\s*###\s*REASONING\s*:\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    pattern_explicit = re.compile(
        r"(?:### REASONING:|\n\s*A:\s*Let's think step by step\.(?:.*\n)*?\s*### REASONING:)\s*(.*?)\s*(?:### FINAL_ANSWER:|\n\s*FINAL_ANSWER:)\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )
    pattern_implicit = re.compile(
        r"(.*?)\s*(?:Answer:|Final Answer:|Final:)\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    match_post = pattern_post.search(generated_text)
    if match_post:
        return match_post.group(2).strip(), match_post.group(1).strip()

    match_explicit = pattern_explicit.search(generated_text)
    if match_explicit:
        return match_explicit.group(1).strip(), match_explicit.group(2).strip()

    match_implicit = pattern_implicit.search(generated_text)
    if match_implicit:
        return match_implicit.group(1).strip(), match_implicit.group(2).strip()

    # Fallback: drop common prompt prefix if present.
    txt = (generated_text or "").strip()
    # If the model echoed the prompt, keep only the tail after the last marker.
    if "### REASONING:" in txt.upper() and txt.upper().rfind("### REASONING:") > 0:
        txt = txt[txt.upper().rfind("### REASONING:") :]
    if "### FINAL_ANSWER:" in txt.upper() and txt.upper().rfind("### FINAL_ANSWER:") > 0:
        txt = txt[txt.upper().rfind("### FINAL_ANSWER:") :]

    lines = txt.splitlines()
    if lines:
        answer = lines[-1].strip()
        reasoning = "\n".join(lines[:-1]).strip()
        if not reasoning:
            return txt.strip(), ""
        return reasoning, answer

    return "", ""
