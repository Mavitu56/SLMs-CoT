import argparse
import json
import os
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _print_kv(title: str, data: Dict[str, Any]) -> None:
    print(title)
    for k, v in data.items():
        print(f"  - {k}: {v}")


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_logits_shards(logits_dir: Path) -> List[Path]:
    return sorted(logits_dir.glob("teacher_logits_shard_*.pt"), key=lambda p: p.name)


def scan_logits_cache(
    logits_dir: Path,
    *,
    max_shards: Optional[int] = None,
    top: int = 10,
    fail_on_nonfinite: bool = False,
    retries: int = 2,
    continue_on_load_error: bool = True,
    use_mmap: bool = True,
    weights_only: bool = True,
    stats_device: str = "cpu",
    gc_every: int = 1,
) -> int:
    try:
        import torch
    except Exception as exc:
        print("Erro: este scanner precisa de 'torch' instalado para ler os .pt.")
        print(f"Detalhe: {exc}")
        return 2

    logits_dir = logits_dir.expanduser().resolve()
    if not logits_dir.exists():
        print(f"Erro: diretório não existe: {logits_dir}")
        return 2

    meta = _try_read_json(logits_dir / "metadata.json")
    if meta:
        _print_kv("metadata.json", {
            "kind": meta.get("kind"),
            "teacher": meta.get("teacher"),
            "tokenizer": meta.get("tokenizer"),
            "tokenizer_hash": meta.get("tokenizer_hash"),
            "max_length": meta.get("max_length"),
            "batch_size": meta.get("batch_size"),
            "split": meta.get("split"),
            "seed": meta.get("seed"),
            "kd_mode": meta.get("kd_mode"),
            "dataset_fingerprint": meta.get("dataset_fingerprint"),
        })

    stats_path = logits_dir / "shard_stats.jsonl"
    if stats_path.exists():
        print(f"Encontrado shard stats: {stats_path}")

    shards = _iter_logits_shards(logits_dir)
    if not shards:
        print(f"Erro: nenhum shard encontrado em: {logits_dir}")
        return 2

    if max_shards is not None:
        shards = shards[: max(0, int(max_shards))]

    total = 0
    total_nonfinite = 0
    load_errors = 0
    load_error_paths: List[Path] = []
    worst: List[Tuple[int, float, int, Path]] = []  # (idx, max_abs, nonfinite, path)

    print(f"Scan logits cache: {logits_dir}")
    print(f"Shards: {len(shards)}")

    for i, shard_path in enumerate(shards):
        payload = None
        last_exc: Optional[BaseException] = None
        for attempt in range(max(1, int(retries) + 1)):
            try:
                load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
                # Newer torch supports weights_only/mmap; older versions will throw TypeError.
                if bool(weights_only):
                    load_kwargs["weights_only"] = True
                if bool(use_mmap):
                    load_kwargs["mmap"] = True
                try:
                    payload = torch.load(shard_path, **load_kwargs)
                except TypeError:
                    # Fallback for older torch.
                    payload = torch.load(shard_path, map_location="cpu")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                # Colab/Drive can intermittently disconnect (Errno 107); retry helps.
                time.sleep(0.5)

        if payload is None:
            load_errors += 1
            load_error_paths.append(shard_path)
            print(f"(ioerr) shard={i} failed_to_load file={shard_path.name} err={type(last_exc).__name__}: {last_exc}")
            if not continue_on_load_error:
                return 2
            continue

        logits = payload.get("logits")
        if logits is None:
            print(f"(warn) shard sem 'logits': {shard_path}")
            continue

        # Keep fp16/bf16 if that's what is on disk to avoid doubling RAM.
        # If there are NaN/Inf, isfinite works fine in fp16.
        t = logits
        if stats_device != "cpu":
            # Optional: move to CUDA for faster reductions (won't reduce RAM; may use VRAM).
            # Use non_blocking only if tensor is pinned (it isn't), but harmless.
            t = t.to(stats_device)

        nonfinite = int((~torch.isfinite(t)).sum().item())
        max_abs = float(t.abs().max().item()) if t.numel() else 0.0
        total += 1
        total_nonfinite += nonfinite
        worst.append((i, max_abs, nonfinite, shard_path))

        if nonfinite:
            print(f"(bad) shard={i} nonfinite={nonfinite} max|logit|={max_abs:.2f} path={shard_path}")
        elif (i + 1) % 50 == 0:
            print(f"(ok) shard={i} max|logit|={max_abs:.2f}")

        # Free memory aggressively (important on Colab with large shards).
        try:
            del t
        except Exception:
            pass
        try:
            del logits
        except Exception:
            pass
        try:
            del payload
        except Exception:
            pass
        if gc_every and ((i + 1) % int(gc_every) == 0):
            gc.collect()
            if stats_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    worst_sorted = sorted(worst, key=lambda t: (t[2] > 0, t[2], t[1]), reverse=True)
    print("Resumo")
    print(f"  shards_lidos: {total}")
    print(f"  nonfinite_total: {total_nonfinite}")
    print(f"  load_errors: {load_errors}")

    if load_error_paths:
        # Show a few to make it actionable.
        show = load_error_paths[: min(10, len(load_error_paths))]
        print(f"  load_error_examples ({len(show)}/{len(load_error_paths)}):")
        for p in show:
            print(f"    - {p}")

    if worst_sorted:
        print(f"Top {min(top, len(worst_sorted))} shards por (nonfinite, max_abs):")
        for idx, max_abs, nonfinite, path in worst_sorted[:top]:
            tag = "BAD" if nonfinite else "OK"
            print(f"  - [{tag}] shard={idx} nonfinite={nonfinite} max|logit|={max_abs:.2f} file={path.name}")

    if fail_on_nonfinite and total_nonfinite > 0:
        return 1
    return 0


def scan_cot_cache(
    cot_file: Path,
    *,
    max_lines: Optional[int] = None,
    fail_on_bad_json: bool = False,
) -> int:
    cot_file = cot_file.expanduser().resolve()
    if not cot_file.exists():
        print(f"Erro: arquivo não existe: {cot_file}")
        return 2

    total = 0
    bad_json = 0
    missing_q = 0
    missing_r = 0
    missing_a = 0
    empty_reasoning = 0
    empty_answer = 0

    print(f"Scan CoT cache: {cot_file}")

    with open(cot_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if max_lines is not None and total >= int(max_lines):
                break
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            q = (rec.get("question") or rec.get("text") or rec.get("prompt") or "").strip()
            r = (rec.get("teacher_reasoning") or "").strip()
            a = (rec.get("teacher_answer") or rec.get("gold_answer") or "").strip()

            if not q:
                missing_q += 1
            if "teacher_reasoning" in rec and not r:
                empty_reasoning += 1
            if "teacher_answer" in rec and not a:
                empty_answer += 1

            # If field absent entirely, count as missing.
            if "teacher_reasoning" not in rec:
                missing_r += 1
            if ("teacher_answer" not in rec) and ("gold_answer" not in rec):
                missing_a += 1

    print("Resumo")
    print(f"  linhas_lidas: {total}")
    print(f"  bad_json: {bad_json}")
    print(f"  missing_question: {missing_q}")
    print(f"  missing_teacher_reasoning_field: {missing_r}")
    print(f"  missing_answer_field: {missing_a}")
    print(f"  empty_teacher_reasoning: {empty_reasoning}")
    print(f"  empty_answer: {empty_answer}")

    if fail_on_bad_json and bad_json > 0:
        return 1
    return 0


def scan_cache_root(cache_root: Path, *, max_shards: Optional[int] = None) -> int:
    cache_root = cache_root.expanduser().resolve()
    if not cache_root.exists():
        print(f"Erro: cache_root não existe: {cache_root}")
        return 2

    logits_dirs = sorted([p for p in cache_root.iterdir() if p.is_dir() and p.name.startswith("teacher_logits_")])
    cot_dirs = sorted([p for p in cache_root.iterdir() if p.is_dir() and p.name.startswith("teacher_cot_")])

    print(f"Cache root: {cache_root}")
    print(f"  logits_dirs: {len(logits_dirs)}")
    print(f"  cot_dirs: {len(cot_dirs)}")

    rc = 0
    for d in logits_dirs:
        out = scan_logits_cache(d, max_shards=max_shards, fail_on_nonfinite=False)
        rc = max(rc, out)

    for d in cot_dirs:
        f = d / "teacher_cot.jsonl"
        if f.exists():
            out = scan_cot_cache(f, max_lines=2000, fail_on_bad_json=False)
            rc = max(rc, out)

    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="Scan caches (teacher logits shards and/or teacher CoT jsonl) for corruption/non-finite values.")
    p.add_argument("--logits_dir", type=str, default=None, help="Diretório teacher_logits_<fp> contendo teacher_logits_shard_*.pt")
    p.add_argument("--cot_file", type=str, default=None, help="Arquivo teacher_cot.jsonl para validar JSON/fields")
    p.add_argument("--cache_root", type=str, default=None, help="Diretório raiz com teacher_logits_* e teacher_cot_*")
    p.add_argument("--max_shards", type=int, default=None, help="Limite de shards (para scan rápido)")
    p.add_argument("--max_lines", type=int, default=None, help="Limite de linhas no CoT jsonl")
    p.add_argument("--top", type=int, default=10, help="Top shards para report")
    p.add_argument("--fail_on_nonfinite", action="store_true", help="Retorna exit code 1 se achar NaN/Inf em logits")
    p.add_argument("--fail_on_bad_json", action="store_true", help="Retorna exit code 1 se achar JSON inválido no CoT")
    p.add_argument("--retries", type=int, default=2, help="Retries ao carregar um shard .pt (útil para Google Drive instável)")
    p.add_argument(
        "--stop_on_load_error",
        action="store_true",
        help="Para no primeiro erro de leitura de shard (por padrão, continua e reporta)",
    )
    p.add_argument(
        "--no_mmap",
        action="store_true",
        help="Desativa torch.load(mmap=True) quando disponível (pode aumentar RAM)",
    )
    p.add_argument(
        "--no_weights_only",
        action="store_true",
        help="Desativa torch.load(weights_only=True) quando disponível",
    )
    p.add_argument(
        "--stats_device",
        type=str,
        default="cpu",
        help="Dispositivo para calcular estatísticas (cpu ou cuda). GPU não reduz RAM; só acelera reduções.",
    )
    p.add_argument(
        "--gc_every",
        type=int,
        default=1,
        help="Faz gc.collect() a cada N shards (reduz pico de RAM no Colab).",
    )

    args = p.parse_args()

    if not args.logits_dir and not args.cot_file and not args.cache_root:
        p.print_help()
        return 2

    rc = 0
    if args.cache_root:
        rc = max(rc, scan_cache_root(Path(args.cache_root), max_shards=args.max_shards))

    if args.logits_dir:
        rc = max(
            rc,
            scan_logits_cache(
                Path(args.logits_dir),
                max_shards=args.max_shards,
                top=int(args.top),
                fail_on_nonfinite=bool(args.fail_on_nonfinite),
                retries=int(args.retries),
                continue_on_load_error=(not bool(args.stop_on_load_error)),
                use_mmap=(not bool(args.no_mmap)),
                weights_only=(not bool(args.no_weights_only)),
                stats_device=str(args.stats_device),
                gc_every=int(args.gc_every),
            ),
        )

    if args.cot_file:
        rc = max(
            rc,
            scan_cot_cache(
                Path(args.cot_file),
                max_lines=args.max_lines,
                fail_on_bad_json=bool(args.fail_on_bad_json),
            ),
        )

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
