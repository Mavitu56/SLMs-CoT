import argparse
import json
import os
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
    worst: List[Tuple[int, float, int, Path]] = []  # (idx, max_abs, nonfinite, path)

    print(f"Scan logits cache: {logits_dir}")
    print(f"Shards: {len(shards)}")

    for i, shard_path in enumerate(shards):
        payload = torch.load(shard_path, map_location="cpu")
        logits = payload.get("logits")
        if logits is None:
            print(f"(warn) shard sem 'logits': {shard_path}")
            continue

        logits_f32 = logits.to(torch.float32)
        nonfinite = int((~torch.isfinite(logits_f32)).sum().item())
        max_abs = float(logits_f32.abs().max().item()) if logits_f32.numel() else 0.0
        total += 1
        total_nonfinite += nonfinite
        worst.append((i, max_abs, nonfinite, shard_path))

        if nonfinite:
            print(f"(bad) shard={i} nonfinite={nonfinite} max|logit|={max_abs:.2f} path={shard_path}")
        elif (i + 1) % 50 == 0:
            print(f"(ok) shard={i} max|logit|={max_abs:.2f}")

    worst_sorted = sorted(worst, key=lambda t: (t[2] > 0, t[2], t[1]), reverse=True)
    print("Resumo")
    print(f"  shards_lidos: {total}")
    print(f"  nonfinite_total: {total_nonfinite}")

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
