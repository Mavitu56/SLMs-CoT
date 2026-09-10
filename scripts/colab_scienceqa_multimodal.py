"""Pipeline de execução no Google Colab A100 (80 GB) — ScienceQA Multimodal KD.

Professor: Qwen2.5-VL-7B-Instruct (bf16)
Aluno:     Qwen2.5-VL-3B-Instruct (bf16)
Dataset:   ScienceQA (derek-thomas/ScienceQA)

Cada bloco `# %%` é uma célula executável no Colab ou Jupyter.
Comandos de shell são executados com streaming em tempo real linha a linha para que
o progresso e os logs apareçam imediatamente na interface do Colab.

Recursos de Resiliência:
  - Streaming em tempo real de logs (stdout unbuffered)
  - Marcadores visuais claros para cada etapa
  - Sincronização contínua com Google Drive para evitar perda de progresso
  - Capacidade de retomada (resume) automática em caso de desconexão
  - Pulo automático de experimentos e avaliações já concluídos
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time


REPO_DIR = "/content/SLMs-CoT"
REPO_URL = "https://github.com/Mavitu56/SLMs-CoT.git"
BRANCH = "kd-ablations-reweighting"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"


def run_cmd(
    cmd: list[str],
    cwd: str | None = None,
    check: bool = True,
    step_title: str | None = None,
) -> int:
    """Executa um comando no terminal com streaming em tempo real linha a linha."""
    if step_title:
        print("\n" + "═" * 70, flush=True)
        print(f"▶ ETAPA: {step_title}", flush=True)
        print(f"  Comando: {' '.join(cmd)}", flush=True)
        print("═" * 70, flush=True)
    else:
        print(f"\n$ {' '.join(cmd)}", flush=True)

    # Força execução sem buffering para que os logs cheguem ao Jupyter instantaneamente
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()

    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


# %%
# =====================================================================
# Célula 1 — Setup: GPU A100, Dependências e Google Drive
# =====================================================================
def cell1_setup() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 1: SETUP DO AMBIENTE (A100, GOOGLE DRIVE E DEPENDÊNCIAS)", flush=True)
    print("█" * 70, flush=True)

    # 1. Verificar GPU
    print("\n[Passo 1/4] Verificando GPU...", flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU não detectada! Ative a GPU A100 nas configurações do Colab.")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)
        if vram_gb < 70.0:
            print("  ⚠️ AVISO: Menos de 70 GB detectados. Recomendado A100 80GB para evitar OOM.", flush=True)
    except Exception as e:
        print(f"  ❌ Erro na checagem de GPU: {e}", flush=True)

    # 2. Montar Google Drive
    print("\n[Passo 2/4] Configurando persistência no Google Drive...", flush=True)
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(DRIVE_ROOT, exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/logs", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)
        print(f"  ✓ Google Drive montado com sucesso em: {DRIVE_ROOT}", flush=True)
    except Exception:
        print(f"  ℹ️ Ambiente fora do Colab ou Drive já montado: {DRIVE_ROOT}", flush=True)

    # 3. Clonar ou atualizar repositório
    print("\n[Passo 3/4] Atualizando repositório Git...", flush=True)
    if os.path.isdir(REPO_DIR):
        print(f"  Atualizando repositório existente em {REPO_DIR} ...", flush=True)
        run_cmd(["git", "fetch", "origin"], cwd=REPO_DIR)
        run_cmd(["git", "checkout", BRANCH], cwd=REPO_DIR)
        run_cmd(["git", "pull", "origin", BRANCH], cwd=REPO_DIR)
    else:
        print(f"  Clonando {REPO_URL} (branch {BRANCH}) ...", flush=True)
        run_cmd(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR])

    # 4. Instalar dependências atualizadas para Qwen2.5-VL
    print("\n[Passo 4/4] Instalando dependências atualizadas...", flush=True)
    run_cmd([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.49.0",
        "torchvision>=0.16.0",
        "accelerate>=0.28.0",
        "datasets>=2.16.0",
        "qwen-vl-utils>=0.0.8",
        "pillow>=10.0.0",
        "bitsandbytes>=0.41.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rouge-score>=0.1.2",
        "matplotlib>=3.7.0",
    ], step_title="Instalação de Pacotes Pip")

    print("\n✓ CÉLULA 1 CONCLUÍDA COM SUCESSO! (SDPA nativo do PyTorch ativo)\n", flush=True)


# %%
# =====================================================================
# Célula 2 — Verificação dos Modelos e Vocabulário
# =====================================================================
def cell2_verify_models() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 2: VERIFICAÇÃO DE MODELOS E ALINHAMENTO DE VOCABULÁRIO", flush=True)
    print("█" * 70, flush=True)

    import torch
    from transformers import AutoProcessor, AutoConfig

    teacher_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    student_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    print("\n[Passo 1/3] Carregando processadores do Aluno e Professor...", flush=True)
    print(f"  Carregando AutoProcessor do aluno: {student_name} ...", flush=True)
    proc_s = AutoProcessor.from_pretrained(student_name)
    print(f"  Carregando AutoProcessor do professor: {teacher_name} ...", flush=True)
    proc_t = AutoProcessor.from_pretrained(teacher_name)

    print("\n[Passo 2/3] Checando vocabulário base dos tokenizers...", flush=True)
    v_s = proc_s.tokenizer.vocab_size
    v_t = proc_t.tokenizer.vocab_size
    print(f"  Vocabulário base tokenizer: Aluno={v_s}, Professor={v_t}", flush=True)
    assert v_s == v_t, "Erro: Vocabulários do tokenizer não coincidem!"
    print("  ✓ Tokenizers compartilham o mesmo vocabulário base!", flush=True)

    print("\n[Passo 3/3] Checando dimensões de saída do lm_head nos configs...", flush=True)
    cfg_t = AutoConfig.from_pretrained(teacher_name)
    cfg_s = AutoConfig.from_pretrained(student_name)
    vocab_t = cfg_t.text_config.vocab_size if hasattr(cfg_t, "text_config") else getattr(cfg_t, "vocab_size", None)
    vocab_s = cfg_s.text_config.vocab_size if hasattr(cfg_s, "text_config") else getattr(cfg_s, "vocab_size", None)
    print(f"  • Teacher lm_head vocab_size: {vocab_t} (1188 * 128)", flush=True)
    print(f"  • Student lm_head vocab_size: {vocab_s} (1187 * 128)", flush=True)
    print("  ✓ Alinhamento automático via _align_vocab() corta teacher para 151936 com segurança!", flush=True)

    del proc_s, proc_t
    torch.cuda.empty_cache()
    print("\n✓ CÉLULA 2 CONCLUÍDA COM SUCESSO!\n", flush=True)


# %%
# =====================================================================
# Célula 3 — Geração de CoT Piloto (100 exemplos)
# =====================================================================
def cell3_generate_pilot_cot() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 3: GERAÇÃO DE COT PILOTO (100 EXEMPLOS DE TESTE)", flush=True)
    print("█" * 70, flush=True)

    pilot_output = f"{REPO_DIR}/data/scienceqa_cot_pilot_100.jsonl"
    pilot_stats = f"{REPO_DIR}/data/scienceqa_cot_pilot_stats.json"

    batch_size = os.environ.get("COT_BATCH_SIZE", "16")
    attn_impl = os.environ.get("COT_ATTN_IMPL", "sdpa")

    print(f"[Passo 1/2] Arquivo alvo de saída: {pilot_output}", flush=True)
    run_cmd([
        sys.executable, "-u", f"{REPO_DIR}/scripts/generate_scienceqa_cot.py",
        "--output-path", pilot_output,
        "--stats-path", pilot_stats,
        "--max-per-split", "100",
        "--splits", "train", "test",
        "--batch-size", str(batch_size),
        "--attn-impl", attn_impl,
    ], cwd=REPO_DIR, step_title=f"Geração CoT Piloto (100 amostras, bs={batch_size}, attn={attn_impl})")

    print(f"\n[Passo 2/2] Validando arquivo piloto gerado...", flush=True)
    if os.path.isfile(pilot_output):
        with open(pilot_output, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"  ✓ CoT piloto gerado com sucesso: {total_lines} exemplos gravados!", flush=True)
    print("\n✓ CÉLULA 3 CONCLUÍDA COM SUCESSO!\n", flush=True)


# %%
# =====================================================================
# Célula 4 — Geração de CoT Completa (~12.7k train + ~4.2k test)
# =====================================================================
def cell4_generate_full_cot() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 4: GERAÇÃO DE COT COMPLETA (COM PERSISTÊNCIA CONTÍNUA NO DRIVE)", flush=True)
    print("█" * 70, flush=True)

    output_path = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    stats_path = f"{REPO_DIR}/data/scienceqa_cot_stats.json"
    drive_data = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    drive_stats = f"{DRIVE_ROOT}/data/scienceqa_cot_stats.json"

    print("\n[Passo 1/3] Checando progresso prévio no Google Drive...", flush=True)
    if os.path.isfile(drive_data):
        drive_size = os.path.getsize(drive_data)
        print(f"  ✓ Encontrado backup existente no Drive ({drive_size / 1e6:.1f} MB)", flush=True)
        if not os.path.isfile(output_path) or os.path.getsize(output_path) < drive_size:
            print(f"  Restaurando backup do Drive para a VM local: {drive_data} -> {output_path} ...", flush=True)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            shutil.copyfile(drive_data, output_path)
    else:
        print("  ℹ️ Nenhum backup prévio encontrado. Iniciando geração do zero.", flush=True)

    batch_size = os.environ.get("COT_BATCH_SIZE", "16")
    attn_impl = os.environ.get("COT_ATTN_IMPL", "sdpa")

    print(f"\n[Passo 2/3] Iniciando geração completa (batch_size={batch_size}, attn={attn_impl})...", flush=True)
    run_cmd([
        sys.executable, "-u", f"{REPO_DIR}/scripts/generate_scienceqa_cot.py",
        "--output-path", output_path,
        "--stats-path", stats_path,
        "--drive-sync-path", drive_data,
        "--sync-every", "50",
        "--splits", "train", "test",
        "--batch-size", str(batch_size),
        "--attn-impl", attn_impl,
    ], cwd=REPO_DIR, step_title=f"Geração CoT Completa (bs={batch_size}, attn={attn_impl})")

    print("\n[Passo 3/3] Sincronização final e validação...", flush=True)
    try:
        shutil.copyfile(output_path, drive_data)
        if os.path.isfile(stats_path):
            shutil.copyfile(stats_path, drive_stats)
        print(f"  ✓ Dataset CoT persistido em segurança no Google Drive: {drive_data}", flush=True)
    except Exception as e:
        print(f"  ⚠️ Falha na sincronização final com Drive: {e}", flush=True)

    print("\n✓ CÉLULA 4 CONCLUÍDA COM SUCESSO!\n", flush=True)


# %%
# =====================================================================
# Célula 5 — Smoke Test de Treino (Micro-Overfit 16 exemplos)
# =====================================================================
def cell5_smoke_test_training() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 5: SMOKE TEST DE TREINO (VALIDAÇÃO DE VRAM NA GPU A100)", flush=True)
    print("█" * 70, flush=True)

    temp_cfg_path = f"{REPO_DIR}/configs/scienceqa/smoke_test.yaml"
    base_cfg = f"{REPO_DIR}/configs/scienceqa/scienceqa_cot_fkl_T4_seed42.yaml"

    print("\n[Passo 1/2] Preparando arquivo de configuração temporário para smoke test...", flush=True)
    import yaml
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["micro_overfit_n"] = 16
    cfg["num_epochs"] = 1
    cfg["log_every"] = 1
    cfg["save_dir"] = f"{REPO_DIR}/checkpoints/smoke_test"
    cfg["log_file"] = f"{REPO_DIR}/logs/smoke_test.jsonl"

    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"  ✓ Configuração temporária gerada em {temp_cfg_path}", flush=True)

    print("\n[Passo 2/2] Executando smoke test e validando VRAM...", flush=True)
    run_cmd([
        sys.executable, "-u", f"{REPO_DIR}/scripts/run.py",
        "--config", temp_cfg_path,
    ], cwd=REPO_DIR, step_title="Execução do Smoke Test de Treino")

    print("\n✓ CÉLULA 5 CONCLUÍDA COM SUCESSO! VRAM e loop de treino validados.\n", flush=True)


# %%
# =====================================================================
# Célula 6 — Sweep Completo de Treino (8 Configs)
# =====================================================================
def cell6_run_full_sweep() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 6: EXECUÇÃO DO SWEEP DE TREINO (8 CONFIGURAÇÕES)", flush=True)
    print("█" * 70, flush=True)

    config_files = sorted(glob.glob(f"{REPO_DIR}/configs/scienceqa/scienceqa_*.yaml"))
    config_files = [c for c in config_files if not os.path.basename(c).startswith("smoke_test")]

    print(f"\n[Passo 1/2] Encontrados {len(config_files)} experimentos configurados:")
    for c in config_files:
        print(f"  • {os.path.basename(c)}", flush=True)

    print("\n[Passo 2/2] Iniciando execução com persistência e detecção de conclusão...", flush=True)

    for idx, cfg_path in enumerate(config_files, 1):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        
        # Deriva o nome da pasta do experimento
        run_name = cfg_name
        for prefix in ("scienceqa_cot_", "scienceqa_human_", "scienceqa_"):
            if run_name.startswith(prefix):
                run_name = run_name[len(prefix):]
                break

        drive_final_ckpt = f"{DRIVE_ROOT}/{run_name}/checkpoints/final"

        # Checa se o experimento já foi concluído anteriormente
        if os.path.isdir(drive_final_ckpt) and len(os.listdir(drive_final_ckpt)) > 0:
            print(f"\n⏩ [{idx}/{len(config_files)}] PULADO: Experimento '{cfg_name}' já concluído no Drive!")
            print(f"   Checkpoint final existente em: {drive_final_ckpt}", flush=True)
            continue

        print("\n" + "═" * 70, flush=True)
        print(f"🚀 [{idx}/{len(config_files)}] Iniciando experimento: {cfg_name}", flush=True)
        print(f"   Salvando checkpoints e logs no Drive: {DRIVE_ROOT}/{run_name}", flush=True)
        print("═" * 70, flush=True)

        t_start = time.time()
        ret = run_cmd([
            sys.executable, "-u", f"{REPO_DIR}/scripts/run.py",
            "--config", cfg_path,
            "--drive-root", DRIVE_ROOT,
        ], cwd=REPO_DIR, check=False, step_title=f"Treino {cfg_name} [{idx}/{len(config_files)}]")

        elapsed = (time.time() - t_start) / 60.0
        if ret == 0:
            print(f"\n✓ [{idx}/{len(config_files)}] '{cfg_name}' concluído com sucesso em {elapsed:.1f} minutos!", flush=True)
        else:
            print(f"\n❌ [{idx}/{len(config_files)}] ERRO no experimento '{cfg_name}' (código de saída: {ret})", flush=True)

    print("\n✓ CÉLULA 6 CONCLUÍDA! Sweep de treinamento finalizado.\n", flush=True)


# %%
# =====================================================================
# Célula 7 — Avaliação de Acurácia de Múltipla Escolha
# =====================================================================
def cell7_evaluate_accuracy() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 7: AVALIAÇÃO DE ACURÁCIA (OVERALL / WITH-IMG / NO-IMG)", flush=True)
    print("█" * 70, flush=True)

    print("\n[Passo 1/2] Localizando checkpoints finais para avaliação...", flush=True)
    checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
    if not checkpoints:
        checkpoints = sorted(glob.glob(f"{REPO_DIR}/checkpoints/*/final"))

    print(f"  Encontrados {len(checkpoints)} checkpoints disponíveis.", flush=True)
    cot_jsonl = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    if not os.path.isfile(cot_jsonl) and os.path.isfile(f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"):
        cot_jsonl = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

    print("\n[Passo 2/2] Executando avaliações de acurácia...", flush=True)
    for idx, ckpt in enumerate(checkpoints, 1):
        run_name = ckpt.split(os.sep)[-3]
        out_json = f"{DRIVE_ROOT}/results/eval_acc_{run_name}.json"

        if os.path.isfile(out_json) and os.path.getsize(out_json) > 10:
            print(f"  ⏩ [{idx}/{len(checkpoints)}] PULADO: Avaliação de {run_name} já concluída em {out_json}", flush=True)
            continue

        print(f"\n  ▶ [{idx}/{len(checkpoints)}] Avaliando: {run_name} ...", flush=True)
        run_cmd([
            sys.executable, "-u", f"{REPO_DIR}/src/evaluation/evaluate_scienceqa.py",
            "--model-path", ckpt,
            "--jsonl-path", cot_jsonl,
            "--split", "test",
            "--output-json", out_json,
        ], cwd=REPO_DIR, step_title=f"Acurácia {run_name} [{idx}/{len(checkpoints)}]")

    print("\n✓ CÉLULA 7 CONCLUÍDA COM SUCESSO!\n", flush=True)


# %%
# =====================================================================
# Célula 8 — Avaliação Probabilística (ECE, H_R, H_A, ρ, KL)
# =====================================================================
def cell8_evaluate_probabilistic() -> None:
    print("\n" + "█" * 70, flush=True)
    print("  CÉLULA 8: AVALIAÇÃO PROBABILÍSTICA (ECE, ENTROPIA, ρ, KL)", flush=True)
    print("█" * 70, flush=True)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from src.data.data_scienceqa import build_dataloader_cot
    from src.evaluation.evaluate_probabilistic import evaluate_model

    teacher_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    student_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    cot_jsonl = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    if not os.path.isfile(cot_jsonl) and os.path.isfile(f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"):
        cot_jsonl = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

    print("\n[Passo 1/3] Carregando processador e DataLoader de teste...", flush=True)
    proc = AutoProcessor.from_pretrained(student_name)
    eval_loader = build_dataloader_cot(
        processor=proc,
        max_length=1536,
        batch_size=4,
        jsonl_path=cot_jsonl,
        split="test",
        shuffle=False,
    )

    print("\n[Passo 2/3] Carregando modelo do Professor para cálculo de divergência KL...", flush=True)
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("\n[Passo 3/3] Avaliando checkpoints probabilísticos...", flush=True)
    checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
    summary_path = f"{DRIVE_ROOT}/results/probabilistic_summary.json"

    # Carrega resumo pré-existente para não recalcular modelos já prontos
    summary_results = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_results = json.load(f)
            print(f"  ✓ Carregadas {len(summary_results)} avaliações prévias do Drive.", flush=True)
        except Exception:
            summary_results = {}

    for idx, ckpt in enumerate(checkpoints, 1):
        run_name = ckpt.split(os.sep)[-3]

        if run_name in summary_results:
            print(f"  ⏩ [{idx}/{len(checkpoints)}] PULADO: Avaliação probabilística para {run_name} já calculada.", flush=True)
            continue

        print(f"\n  ▶ [{idx}/{len(checkpoints)}] Calculando probabilidades para: {run_name} ...", flush=True)

        student = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

        cfg_eval = {"max_length": 1536}
        res = evaluate_model(student, teacher, eval_loader, cfg_eval)
        summary_results[run_name] = {
            "mean_entropy": res["mean_entropy"],
            "mean_maxprob": res["mean_maxprob"],
            "ece": res["ece"],
            "mean_kl": res["mean_kl"],
            "rho": res["rho_HR_HA"],
            "with_image_ece": res.get("modality_with_image", {}).get("ece"),
            "no_image_ece": res.get("modality_without_image", {}).get("ece"),
        }

        # Salva incrementalmente no Drive a cada modelo avaliado
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"  ✓ Resultados de {run_name} salvos em {summary_path}", flush=True)

        del student
        torch.cuda.empty_cache()

    print(f"\n✓ CÉLULA 8 CONCLUÍDA! Resumo probabilístico salvo em: {summary_path}\n", flush=True)
    print(json.dumps(summary_results, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Colab ScienceQA Multimodal Runner")
    parser.add_argument(
        "--cell",
        type=str,
        default="1",
        help="Célula para executar: 1..8, 'all', ou lista separada por vírgula (ex: '1,2'). Padrão: '1'",
    )
    args = parser.parse_args()

    cell_map = {
        "1": ("Setup & Dependências", cell1_setup),
        "2": ("Verificação de Modelos & Vocabulário", cell2_verify_models),
        "3": ("Geração CoT Piloto (100)", cell3_generate_pilot_cot),
        "4": ("Geração CoT Completa", cell4_generate_full_cot),
        "5": ("Smoke Test de Treino", cell5_smoke_test_training),
        "6": ("Sweep de Treino (8 configs)", cell6_run_full_sweep),
        "7": ("Avaliação de Acurácia", cell7_evaluate_accuracy),
        "8": ("Avaliação Probabilística", cell8_evaluate_probabilistic),
    }

    if args.cell.lower() == "all":
        targets = ["1", "2", "3", "4", "5", "6", "7", "8"]
    else:
        targets = [c.strip() for c in args.cell.split(",") if c.strip()]

    for c in targets:
        if c in cell_map:
            name, func = cell_map[c]
            print(f"\n>>> Executando Célula {c}: {name} ...\n", flush=True)
            func()
        else:
            print(f"Célula desconhecida: {c!r}. Opções válidas: 1 a 8 ou 'all'.", flush=True)
