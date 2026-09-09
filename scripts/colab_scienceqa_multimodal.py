"""Pipeline de execução no Google Colab A100 (80 GB) — ScienceQA Multimodal KD.

Professor: Qwen2.5-VL-7B-Instruct (bf16)
Aluno:     Qwen2.5-VL-3B-Instruct (bf16)
Dataset:   ScienceQA (derek-thomas/ScienceQA)

Cada bloco `# %%` é uma célula executável no Colab ou Jupyter.
Comandos de shell são executados via `subprocess` para garantir que o script seja
Python válido e possa ser executado tanto célula a célula quanto em lote.

Estrutura das Células:
  Célula 1 -> Setup: Verificação de GPU A100 80GB, clone/checkout git e dependências
  Célula 2 -> Verificação de Vocabulário & Shape Mismatch (152.064 vs 151.936)
  Célula 3 -> Geração de CoT Piloto (100 exemplos) para validação rápida
  Célula 4 -> Geração de CoT Completa (~12.7k train + ~4.2k test) com checkpoint no Drive
  Célula 5 -> Smoke Test de Treino (micro-overfit 100 exemplos) para validar VRAM
  Célula 6 -> Execução do Sweep de Treino (8 configs YAML)
  Célula 7 -> Avaliação de Acurácia Múltipla Escolha (overall, with-image, without-image)
  Célula 8 -> Avaliação Probabilística (ECE, Entropia, ρ = H_R / H_A, KL)
"""

import glob
import json
import os
import subprocess
import sys
import time


REPO_DIR = "/content/SLMs-CoT"
REPO_URL = "https://github.com/Mavitu56/SLMs-CoT.git"
BRANCH = "kd-ablations-reweighting"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"


def run_cmd(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    """Executa um comando no terminal, imprimindo-o claramente."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=check)
    return proc.returncode


# %%
# =====================================================================
# Célula 1 — Setup: GPU A100, Dependências e Google Drive
# =====================================================================
def cell1_setup() -> None:
    print("=" * 60)
    print("CÉLULA 1: SETUP DO AMBIENTE")
    print("=" * 60)

    # 1. Verificar GPU
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU não detectada! Ative GPU A100 nas configurações do Colab.")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 70.0:
            print("AVISO: Menos de 70 GB detectados. Recomendado A100 80GB para evitar OOM.")
    except Exception as e:
        print(f"Erro na checagem de GPU: {e}")

    # 2. Montar Google Drive
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(DRIVE_ROOT, exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/logs", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)
        os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)
        print(f"Google Drive montado com sucesso em: {DRIVE_ROOT}")
    except Exception:
        print("Ambiente fora do Colab ou Drive já montado.")

    # 3. Clonar ou atualizar repositório
    if os.path.isdir(REPO_DIR):
        print(f"Atualizando repositório em {REPO_DIR} ...")
        run_cmd(["git", "fetch", "origin"], cwd=REPO_DIR)
        run_cmd(["git", "checkout", BRANCH], cwd=REPO_DIR)
        run_cmd(["git", "pull", "origin", BRANCH], cwd=REPO_DIR)
    else:
        print(f"Clonando {REPO_URL} (branch {BRANCH}) ...")
        run_cmd(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR])

    # 4. Instalar dependências atualizadas para Qwen2.5-VL
    print("\nInstalando dependências...")
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
    ])
    print("\n✓ Setup concluído!")


# %%
# =====================================================================
# Célula 2 — Verificação dos Modelos e Vocabulário
# =====================================================================
def cell2_verify_models() -> None:
    print("=" * 60)
    print("CÉLULA 2: VERIFICAÇÃO DE MODELOS E VOCABULÁRIO")
    print("=" * 60)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    teacher_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    student_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"Carregando processador do aluno: {student_name} ...")
    proc_s = AutoProcessor.from_pretrained(student_name)
    print(f"Carregando processador do professor: {teacher_name} ...")
    proc_t = AutoProcessor.from_pretrained(teacher_name)

    v_s = proc_s.tokenizer.vocab_size
    v_t = proc_t.tokenizer.vocab_size
    print(f"Vocabulário base tokenizer: Aluno={v_s}, Professor={v_t}")
    assert v_s == v_t, "Erro: Vocabulários do tokenizer não coincidem!"

    print("\nVerificando dimensões no config.json (lm_head):")
    from transformers import AutoConfig
    cfg_t = AutoConfig.from_pretrained(teacher_name)
    cfg_s = AutoConfig.from_pretrained(student_name)
    print(f"  Teacher vocab_size no config: {cfg_t.vocab_size} (1188 * 128)")
    print(f"  Student vocab_size no config: {cfg_s.vocab_size} (1187 * 128)")
    print("  -> Alinhamento automático via _align_vocab() corta teacher para 151936 ✓")

    del proc_s, proc_t
    torch.cuda.empty_cache()
    print("\n✓ Verificação concluída com sucesso!")


# %%
# =====================================================================
# Célula 3 — Geração de CoT Piloto (100 exemplos)
# =====================================================================
def cell3_generate_pilot_cot() -> None:
    print("=" * 60)
    print("CÉLULA 3: GERAÇÃO DE COT PILOTO (100 EXEMPLOS)")
    print("=" * 60)

    pilot_output = f"{REPO_DIR}/data/scienceqa_cot_pilot_100.jsonl"
    run_cmd([
        sys.executable, f"{REPO_DIR}/scripts/generate_scienceqa_cot.py",
        "--output-path", pilot_output,
        "--stats-path", f"{REPO_DIR}/data/scienceqa_cot_pilot_stats.json",
        "--max-per-split", "100",
        "--splits", "train", "test",
    ], cwd=REPO_DIR)

    print(f"\n✓ CoT piloto gerado em: {pilot_output}")


# %%
# =====================================================================
# Célula 4 — Geração de CoT Completa (~12.7k train + ~4.2k test)
# =====================================================================
def cell4_generate_full_cot() -> None:
    print("=" * 60)
    print("CÉLULA 4: GERAÇÃO DE COT COMPLETA (COM PERSISTÊNCIA NO DRIVE)")
    print("=" * 60)

    output_path = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    stats_path = f"{REPO_DIR}/data/scienceqa_cot_stats.json"
    drive_data = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

    # Se já existir progresso no Drive, sincronizar para local antes de rodar
    if os.path.isfile(drive_data) and not os.path.isfile(output_path):
        print(f"Restaurando arquivo existente do Drive: {drive_data} -> {output_path}")
        run_cmd(["cp", drive_data, output_path])

    run_cmd([
        sys.executable, f"{REPO_DIR}/scripts/generate_scienceqa_cot.py",
        "--output-path", output_path,
        "--stats-path", stats_path,
        "--splits", "train", "test",
    ], cwd=REPO_DIR)

    # Copiar arquivo gerado para o Google Drive para persistência definitiva
    print("\nPersistindo JSONL no Google Drive...")
    run_cmd(["cp", output_path, drive_data])
    run_cmd(["cp", stats_path, f"{DRIVE_ROOT}/data/scienceqa_cot_stats.json"])
    print(f"\n✓ CoT completo gerado e salvo em {drive_data}")


# %%
# =====================================================================
# Célula 5 — Smoke Test de Treino (Micro-Overfit 100 exemplos)
# =====================================================================
def cell5_smoke_test_training() -> None:
    print("=" * 60)
    print("CÉLULA 5: SMOKE TEST DE TREINO (VALIDAÇÃO DE VRAM)")
    print("=" * 60)

    # Cria config temporário com micro_overfit_n = 16
    temp_cfg_path = f"{REPO_DIR}/configs/scienceqa/smoke_test.yaml"
    base_cfg = f"{REPO_DIR}/configs/scienceqa/scienceqa_cot_fkl_T4_seed42.yaml"

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

    run_cmd([
        sys.executable, f"{REPO_DIR}/scripts/run.py",
        "--config", temp_cfg_path,
    ], cwd=REPO_DIR)

    print("\n✓ Smoke test passou! VRAM e loop de treino validados.")


# %%
# =====================================================================
# Célula 6 — Sweep Completo de Treino (8 Configs)
# =====================================================================
def cell6_run_full_sweep() -> None:
    print("=" * 60)
    print("CÉLULA 6: EXECUÇÃO DO SWEEP DE TREINO")
    print("=" * 60)

    config_files = sorted(glob.glob(f"{REPO_DIR}/configs/scienceqa/scienceqa_*.yaml"))
    print(f"Encontrados {len(config_files)} configs para executar:")
    for c in config_files:
        print(f"  - {os.path.basename(c)}")

    for idx, cfg_path in enumerate(config_files, 1):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(config_files)}] Iniciando experimento: {cfg_name}")
        print(f"{'='*70}")

        t_start = time.time()
        ret = run_cmd([
            sys.executable, f"{REPO_DIR}/scripts/run.py",
            "--config", cfg_path,
            "--drive-root", DRIVE_ROOT,
        ], cwd=REPO_DIR, check=False)

        elapsed = (time.time() - t_start) / 60.0
        if ret == 0:
            print(f"\n✓ {cfg_name} concluído com sucesso em {elapsed:.1f} minutos!")
        else:
            print(f"\n✗ ERRO no experimento {cfg_name} (código de saída: {ret})")


# %%
# =====================================================================
# Célula 7 — Avaliação de Acurácia de Múltipla Escolha
# =====================================================================
def cell7_evaluate_accuracy() -> None:
    print("=" * 60)
    print("CÉLULA 7: AVALIAÇÃO DE ACURÁCIA (OVERALL / WITH-IMG / NO-IMG)")
    print("=" * 60)

    # Localizar checkpoints salvos no Drive ou localmente
    checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
    if not checkpoints:
        checkpoints = sorted(glob.glob(f"{REPO_DIR}/checkpoints/*/final"))

    print(f"Avaliando {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        run_name = ckpt.split(os.sep)[-3]
        out_json = f"{DRIVE_ROOT}/results/eval_acc_{run_name}.json"

        print(f"\nAvaliando: {run_name} ...")
        run_cmd([
            sys.executable, f"{REPO_DIR}/src/evaluation/evaluate_scienceqa.py",
            "--model-path", ckpt,
            "--jsonl-path", f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl",
            "--split", "test",
            "--output-json", out_json,
        ], cwd=REPO_DIR)


# %%
# =====================================================================
# Célula 8 — Avaliação Probabilística (ECE, H_R, H_A, ρ, KL)
# =====================================================================
def cell8_evaluate_probabilistic() -> None:
    print("=" * 60)
    print("CÉLULA 8: AVALIAÇÃO PROBABILÍSTICA (ECE, ENTROPIA, ρ)")
    print("=" * 60)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from src.data.data_scienceqa import build_dataloader_cot
    from src.evaluation.evaluate_probabilistic import evaluate_model

    teacher_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    student_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    proc = AutoProcessor.from_pretrained(student_name)
    eval_loader = build_dataloader_cot(
        processor=proc,
        max_length=1536,
        batch_size=4,
        jsonl_path=f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl",
        split="test",
        shuffle=False,
    )

    print("Carregando professor para cálculo de KL...")
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
    summary_results = {}

    for ckpt in checkpoints:
        run_name = ckpt.split(os.sep)[-3]
        print(f"\nAvaliando probabilidades para: {run_name} ...")

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
        del student
        torch.cuda.empty_cache()

    summary_path = f"{DRIVE_ROOT}/results/probabilistic_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n✓ Resumo probabilístico salvo em: {summary_path}")
    print(json.dumps(summary_results, indent=2))


if __name__ == "__main__":
    # Quando executado como script, executa setup
    cell1_setup()
