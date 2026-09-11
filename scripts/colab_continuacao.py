"""Pipeline de continuação no Google Colab A100 — ScienceQA Multimodal KD.

A geração de CoT (Fase 1.1) já foi concluída (16.965 exemplos).
Este script faz a recuperação dos dados e executa as fases restantes:
  - Célula A: Montar Drive + salvar/restaurar dados de CoT
  - Célula B: Instalar dependências
  - Célula C: Smoke test de treino (16 exemplos, 1 época)
  - Célula D: Sweep completo de treino (8 configs)
  - Célula E: Avaliação de acurácia (multiple-choice)
  - Célula F: Avaliação probabilística (ECE, entropia, KL, ρ)

Ordem de execução: A → B → C → D → E → F

Se o Colab desconectar, reconecte, re-execute A e B, e continue de onde parou.
Todas as células detectam trabalho já concluído e pulam automaticamente.

INSTRUÇÕES:
  Se os arquivos gerados (scienceqa_cot_qwen25_vl_7b.jsonl e
  scienceqa_cot_stats.json) já foram baixados para sua máquina local,
  faça upload manual para o Google Drive ANTES de rodar a Célula A:

    Google Drive/SLM_ScienceQA_Multimodal/data/
      ├── scienceqa_cot_qwen25_vl_7b.jsonl
      └── scienceqa_cot_stats.json
"""

# %%
# =====================================================================
# CÉLULA A — Montar Google Drive e Salvar/Restaurar Dados de CoT
# =====================================================================
import os
import shutil
import json

REPO_DIR = "/content/SLMs-CoT"
REPO_URL = "https://github.com/Mavitu56/SLMs-CoT.git"
BRANCH = "kd-ablations-reweighting"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA A: MONTAR DRIVE + SALVAR/RESTAURAR DADOS DE CoT")
print("█" * 70)

# --- 1. Montar Google Drive ---
print("\n[1/5] Montando Google Drive...", flush=True)
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")

for subdir in ["data", "checkpoints", "logs", "results"]:
    os.makedirs(f"{DRIVE_ROOT}/{subdir}", exist_ok=True)
print(f"  ✓ Drive montado. Pasta raiz: {DRIVE_ROOT}")

# --- 2. Procurar arquivos de CoT em todos os locais possíveis ---
print("\n[2/5] Procurando arquivos de CoT gerados...", flush=True)

JSONL_NAME = "scienceqa_cot_qwen25_vl_7b.jsonl"
STATS_NAME = "scienceqa_cot_stats.json"

search_paths = [
    f"{REPO_DIR}/data/{JSONL_NAME}",
    f"{REPO_DIR}/{JSONL_NAME}",
    f"/content/{JSONL_NAME}",
    f"{DRIVE_ROOT}/data/{JSONL_NAME}",
]

jsonl_found = None
for p in search_paths:
    if os.path.isfile(p):
        size_mb = os.path.getsize(p) / 1e6
        print(f"  ✓ ENCONTRADO: {p} ({size_mb:.1f} MB)")
        if jsonl_found is None or os.path.getsize(p) > os.path.getsize(jsonl_found):
            jsonl_found = p

stats_found = None
for p in [sp.replace(JSONL_NAME, STATS_NAME) for sp in search_paths]:
    if os.path.isfile(p):
        print(f"  ✓ Stats encontrado: {p}")
        stats_found = p
        break

if jsonl_found is None:
    print("\n  ❌ NENHUM ARQUIVO JSONL ENCONTRADO!")
    print("     Faça upload manual de 'scienceqa_cot_qwen25_vl_7b.jsonl'")
    print(f"     para: {DRIVE_ROOT}/data/")
    print("     Depois re-execute esta célula.")
    raise FileNotFoundError(
        f"JSONL não encontrado. Faça upload para {DRIVE_ROOT}/data/{JSONL_NAME}"
    )

# --- 3. Salvar no Drive (se ainda não está lá) ---
print("\n[3/5] Salvando no Google Drive...", flush=True)
drive_jsonl = f"{DRIVE_ROOT}/data/{JSONL_NAME}"
drive_stats = f"{DRIVE_ROOT}/data/{STATS_NAME}"

if jsonl_found != drive_jsonl:
    shutil.copyfile(jsonl_found, drive_jsonl)
    print(f"  ✓ JSONL salvo no Drive: {drive_jsonl}")
else:
    print(f"  ✓ JSONL já estava no Drive")

if stats_found and stats_found != drive_stats:
    shutil.copyfile(stats_found, drive_stats)
    print(f"  ✓ Stats salvo no Drive: {drive_stats}")

# --- 4. Clonar/atualizar repositório ---
print("\n[4/5] Atualizando repositório...", flush=True)
if os.path.isdir(REPO_DIR):
    os.system(f"cd {REPO_DIR} && git fetch origin && git checkout {BRANCH} && git pull origin {BRANCH}")
else:
    os.system(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}")

# --- 5. Colocar dados no local correto para o treino ---
print("\n[5/5] Colocando dados no local correto do repo...", flush=True)
repo_data_dir = f"{REPO_DIR}/data"
os.makedirs(repo_data_dir, exist_ok=True)

repo_jsonl = f"{repo_data_dir}/{JSONL_NAME}"
repo_stats = f"{repo_data_dir}/{STATS_NAME}"

if not os.path.isfile(repo_jsonl) or os.path.getsize(repo_jsonl) < os.path.getsize(drive_jsonl):
    shutil.copyfile(drive_jsonl, repo_jsonl)
    print(f"  ✓ Copiado Drive → {repo_jsonl}")
else:
    print(f"  ✓ JSONL já está em {repo_jsonl}")

if os.path.isfile(drive_stats):
    if not os.path.isfile(repo_stats) or os.path.getsize(repo_stats) < os.path.getsize(drive_stats):
        shutil.copyfile(drive_stats, repo_stats)

# --- Validação final ---
with open(repo_jsonl, "r", encoding="utf-8") as f:
    n_lines = sum(1 for line in f if line.strip())

print(f"\n{'═' * 70}")
print(f"✅ VALIDAÇÃO: {n_lines} exemplos de CoT prontos em {repo_jsonl}")
print(f"   Drive backup seguro em: {drive_jsonl}")

if os.path.isfile(repo_stats):
    with open(repo_stats, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for split_name, s in stats.items():
        acc = s["accuracy_overall"] * 100
        n = s["n_total"]
        n_img = s["n_with_image"]
        print(f"   {split_name}: {n} exemplos ({n_img} com imagem), acurácia={acc:.1f}%")

print(f"{'═' * 70}")
print("\n✓ CÉLULA A CONCLUÍDA COM SUCESSO!")


# %%
# =====================================================================
# CÉLULA B — Instalar Dependências
# =====================================================================
import subprocess
import sys
import torch

print("█" * 70)
print("  CÉLULA B: INSTALAÇÃO DE DEPENDÊNCIAS")
print("█" * 70)

subprocess.check_call([
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

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\n✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
print("✓ CÉLULA B CONCLUÍDA!")


# %%
# =====================================================================
# CÉLULA C — Smoke Test de Treino (16 exemplos, 1 época)
# =====================================================================
#
# Teste rápido (~2-5 min) para validar que o treino funciona na A100.
# Se passar sem OOM, o sweep completo vai funcionar.
#
import os
import sys
import yaml

REPO_DIR = "/content/SLMs-CoT"

print("█" * 70)
print("  CÉLULA C: SMOKE TEST DE TREINO (16 EXEMPLOS, 1 ÉPOCA)")
print("█" * 70)

base_cfg_path = f"{REPO_DIR}/configs/scienceqa/scienceqa_cot_fkl_T4_seed42.yaml"
temp_cfg_path = f"{REPO_DIR}/configs/scienceqa/smoke_test.yaml"

with open(base_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["micro_overfit_n"] = 16
cfg["num_epochs"] = 1
cfg["log_every"] = 1
cfg["save_dir"] = f"{REPO_DIR}/checkpoints/smoke_test"
cfg["log_file"] = f"{REPO_DIR}/logs/smoke_test.jsonl"

os.makedirs(f"{REPO_DIR}/logs", exist_ok=True)

with open(temp_cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f)

print(f"  Config: {temp_cfg_path}")
print("  Executando smoke test...\n", flush=True)

ret = os.system(f"cd {REPO_DIR} && {sys.executable} -u scripts/run.py --config {temp_cfg_path}")

if ret == 0:
    print("\n✓ CÉLULA C CONCLUÍDA! Treino validado na A100.")
else:
    print(f"\n❌ Smoke test falhou com código {ret}. Verifique os logs acima.")


# %%
# =====================================================================
# CÉLULA D — Sweep Completo de Treino (8 Configurações)
#
# Cada experimento salva checkpoints e logs DIRETAMENTE no Google Drive.
# Se o Colab desconectar, reconecte, re-execute A e B, e rode esta
# célula novamente — experimentos já concluídos serão pulados.
#
# Tempo estimado: ~2-6h (dependendo da GPU e número de experimentos)
# =====================================================================
import os
import sys
import glob
import time

REPO_DIR = "/content/SLMs-CoT"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA D: SWEEP DE TREINO (8 CONFIGURAÇÕES → GOOGLE DRIVE)")
print("█" * 70)

config_files = sorted(glob.glob(f"{REPO_DIR}/configs/scienceqa/scienceqa_*.yaml"))
config_files = [c for c in config_files if "smoke_test" not in os.path.basename(c)]

print(f"\n{len(config_files)} experimentos encontrados:")
for c in config_files:
    print(f"  • {os.path.basename(c)}")

print(f"\nCheckpoints e logs serão salvos em: {DRIVE_ROOT}/")
print("Experimentos já concluídos serão automaticamente pulados.\n")

for idx, cfg_path in enumerate(config_files, 1):
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]

    # Derivar nome do run (remove prefixo scienceqa_cot_ etc.)
    run_name = cfg_name
    for prefix in ("scienceqa_cot_", "scienceqa_human_", "scienceqa_"):
        if run_name.startswith(prefix):
            run_name = run_name[len(prefix):]
            break

    drive_final = f"{DRIVE_ROOT}/{run_name}/checkpoints/final"

    # Pular se já concluído
    if os.path.isdir(drive_final) and len(os.listdir(drive_final)) > 0:
        print(f"⏩ [{idx}/{len(config_files)}] PULADO: {cfg_name} (já concluído no Drive)")
        continue

    print(f"\n{'═' * 70}")
    print(f"🚀 [{idx}/{len(config_files)}] {cfg_name}")
    print(f"   Salvando em: {DRIVE_ROOT}/{run_name}/")
    print(f"{'═' * 70}\n")

    t0 = time.time()
    ret = os.system(
        f"cd {REPO_DIR} && {sys.executable} -u scripts/run.py "
        f"--config {cfg_path} "
        f"--drive-root {DRIVE_ROOT}"
    )
    elapsed = (time.time() - t0) / 60.0

    if ret == 0:
        print(f"\n✓ [{idx}/{len(config_files)}] {cfg_name} concluído em {elapsed:.1f} min")
    else:
        print(f"\n❌ [{idx}/{len(config_files)}] ERRO em {cfg_name} (código: {ret})")

print("\n" + "═" * 70)
print("✓ CÉLULA D CONCLUÍDA! Sweep de treinamento finalizado.")
print("═" * 70)


# %%
# =====================================================================
# CÉLULA E — Avaliação de Acurácia nos Checkpoints (Multiple-Choice)
# =====================================================================
import os
import sys
import glob
import json

REPO_DIR = "/content/SLMs-CoT"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA E: AVALIAÇÃO DE ACURÁCIA (OVERALL / WITH-IMG / NO-IMG)")
print("█" * 70)

# Encontrar checkpoints finais no Drive
checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
print(f"\n{len(checkpoints)} checkpoints encontrados para avaliar.")

if len(checkpoints) == 0:
    print("  ⚠️ Nenhum checkpoint encontrado. Execute a Célula D primeiro.")
else:
    cot_jsonl = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    if not os.path.isfile(cot_jsonl):
        cot_jsonl = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

    os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)

    for idx, ckpt in enumerate(checkpoints, 1):
        run_name = ckpt.split(os.sep)[-3]
        out_json = f"{DRIVE_ROOT}/results/eval_acc_{run_name}.json"

        if os.path.isfile(out_json) and os.path.getsize(out_json) > 10:
            print(f"  ⏩ [{idx}/{len(checkpoints)}] PULADO: {run_name} (já avaliado)")
            continue

        print(f"\n  ▶ [{idx}/{len(checkpoints)}] Avaliando: {run_name}")
        ret = os.system(
            f"cd {REPO_DIR} && {sys.executable} -u src/evaluation/evaluate_scienceqa.py "
            f"--model-path {ckpt} "
            f"--jsonl-path {cot_jsonl} "
            f"--split test "
            f"--output-json {out_json}"
        )
        if ret != 0:
            print(f"    ❌ Erro na avaliação de {run_name}")

    # --- Resumo dos resultados ---
    print(f"\n{'═' * 70}")
    print("📊 RESUMO DE ACURÁCIA")
    print(f"{'═' * 70}")

    result_files = sorted(glob.glob(f"{DRIVE_ROOT}/results/eval_acc_*.json"))
    for rf in result_files:
        name = os.path.basename(rf).replace("eval_acc_", "").replace(".json", "")
        try:
            with open(rf, "r", encoding="utf-8") as fh:
                r = json.load(fh)
            acc = r.get("accuracy_overall", r.get("accuracy", 0)) * 100
            acc_img = r.get("accuracy_with_image", 0)
            acc_noimg = r.get("accuracy_without_image", 0)
            if acc_img:
                acc_img *= 100
            if acc_noimg:
                acc_noimg *= 100
            line = f"  {name:40s} → overall={acc:.1f}%"
            if acc_img:
                line += f"  img={acc_img:.1f}%  noimg={acc_noimg:.1f}%"
            print(line)
        except Exception as e:
            print(f"  {name:40s} → erro ao ler: {e}")

    print(f"{'═' * 70}")

print("\n✓ CÉLULA E CONCLUÍDA!")


# %%
# =====================================================================
# CÉLULA F — Avaliação Probabilística (ECE, Entropia, KL, ρ)
# =====================================================================
import os
import sys
import glob
import json

REPO_DIR = "/content/SLMs-CoT"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA F: AVALIAÇÃO PROBABILÍSTICA (ECE, ENTROPIA, KL, ρ)")
print("█" * 70)

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.data.data_scienceqa import build_dataloader_cot
from src.evaluation.evaluate_probabilistic import evaluate_model

teacher_name = "Qwen/Qwen2.5-VL-7B-Instruct"
student_name = "Qwen/Qwen2.5-VL-3B-Instruct"

cot_jsonl = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
if not os.path.isfile(cot_jsonl):
    cot_jsonl = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

# 1. DataLoader de teste
print("\n[1/3] Carregando processador e DataLoader de teste...", flush=True)
proc = AutoProcessor.from_pretrained(student_name)
eval_loader = build_dataloader_cot(
    processor=proc,
    max_length=1536,
    batch_size=4,
    jsonl_path=cot_jsonl,
    split="test",
    shuffle=False,
)

# 2. Carregar professor
print("\n[2/3] Carregando modelo do Professor...", flush=True)
teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
for p in teacher.parameters():
    p.requires_grad = False

# 3. Avaliar cada checkpoint
print("\n[3/3] Avaliando checkpoints...", flush=True)
checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
summary_path = f"{DRIVE_ROOT}/results/probabilistic_summary.json"

summary_results = {}
if os.path.isfile(summary_path):
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_results = json.load(f)
        print(f"  ✓ Carregadas {len(summary_results)} avaliações prévias.", flush=True)
    except Exception:
        summary_results = {}

for idx, ckpt in enumerate(checkpoints, 1):
    run_name = ckpt.split(os.sep)[-3]

    if run_name in summary_results:
        print(f"  ⏩ [{idx}/{len(checkpoints)}] PULADO: {run_name} (já calculado)")
        continue

    print(f"\n  ▶ [{idx}/{len(checkpoints)}] Calculando: {run_name} ...", flush=True)

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

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"  ✓ Resultados de {run_name} salvos.", flush=True)

    del student
    torch.cuda.empty_cache()

# --- Resumo ---
print(f"\n{'═' * 70}")
print("📊 RESUMO PROBABILÍSTICO")
print(f"{'═' * 70}")
print(json.dumps(summary_results, indent=2))
print(f"{'═' * 70}")
print(f"\nResultados salvos em: {summary_path}")
print("\n✓ CÉLULA F CONCLUÍDA!")
