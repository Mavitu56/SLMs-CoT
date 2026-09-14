# =====================================================================
# CÉLULA F — Avaliação Probabilística (ECE, Entropia, KL, ρ)
#
# Copie e cole TODO este conteúdo dentro da Célula F no Google Colab.
#
# RECURSOS:
#   - Logs a cada 25 batches com tokens avaliados, KL, Entropia e ETA
#   - Batch size 8 para forward pass rápido (aluno 3B + professor 7B em bf16)
#   - Pula automaticamente checkpoints que já foram avaliados
#   - Salva incrementalmente no Google Drive após cada checkpoint
#   - Tabela comparativa final consolidada de calibração e incerteza
# =====================================================================
import os
import sys
import glob
import json
import time

REPO_DIR = "/content/SLMs-CoT"
BRANCH = "kd-ablations-reweighting"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA F: AVALIAÇÃO PROBABILÍSTICA (ECE, ENTROPIA, KL, ρ)")
print("█" * 70)

# Sincronizar repositório com o GitHub
print("\n[Passo 1/4] Atualizando código do repositório...", flush=True)
os.system(f"cd {REPO_DIR} && git fetch origin && git checkout {BRANCH} && git pull origin {BRANCH}")

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
BATCH_SIZE = 8  # Forward pass rápido com aluno 3B + professor 7B
print(f"\n[Passo 2/4] Carregando DataLoader de teste (batch_size={BATCH_SIZE})...", flush=True)
proc = AutoProcessor.from_pretrained(student_name)
eval_loader = build_dataloader_cot(
    processor=proc,
    max_length=1536,
    batch_size=BATCH_SIZE,
    jsonl_path=cot_jsonl,
    split="test",
    shuffle=False,
)
print(f"  ✓ DataLoader pronto ({len(eval_loader)} batches)", flush=True)

# 2. Carregar modelo do Professor (7B)
print(f"\n[Passo 3/4] Carregando Professor ({teacher_name}) em bf16 + SDPA...", flush=True)
teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    teacher_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
).eval()
for p in teacher.parameters():
    p.requires_grad = False

vram_teacher = torch.cuda.memory_allocated() / (1024**3)
print(f"  ✓ Professor carregado na GPU (VRAM usada: {vram_teacher:.1f} GB)", flush=True)

# 3. Localizar Checkpoints e Avaliar
print("\n[Passo 4/4] Buscando checkpoints no Google Drive...", flush=True)
checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
summary_path = f"{DRIVE_ROOT}/results/probabilistic_summary.json"

summary_results = {}
if os.path.isfile(summary_path):
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_results = json.load(f)
        print(f"  ✓ Carregadas {len(summary_results)} avaliações prévias do Drive.")
    except Exception:
        summary_results = {}

pending = []
for ckpt in checkpoints:
    norm_p = os.path.normpath(ckpt)
    r_name = os.path.basename(os.path.dirname(os.path.dirname(norm_p)))
    if r_name in summary_results:
        print(f"  ⏩ PULADO: {r_name} (já calculado)")
    else:
        pending.append((ckpt, r_name))

print(f"\nIniciando avaliação de {len(pending)} checkpoints pendentes...\n", flush=True)
eval_start = time.time()

for idx, (ckpt, run_name) in enumerate(pending, 1):
    print(f"{'═' * 70}")
    print(f"▶ [{idx}/{len(pending)}] Avaliando Calibração e Incerteza: {run_name}")
    print(f"  Origem: {ckpt}")
    print(f"{'═' * 70}", flush=True)

    t0 = time.time()

    student = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
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

    elapsed = (time.time() - t0) / 60.0
    ece_val = res.get("ece")
    ece_img = res.get("modality_with_image", {}).get("ece") if res.get("modality_with_image") else None
    ece_noimg = res.get("modality_without_image", {}).get("ece") if res.get("modality_without_image") else None
    kl_val = res.get("mean_kl")
    ent_val = res.get("mean_entropy")
    rho_val = res.get("rho_HR_HA")

    s_ece = f"{ece_val:.4f}" if ece_val is not None else "N/A"
    s_ece_img = f"{ece_img:.4f}" if ece_img is not None else "N/A"
    s_ece_noimg = f"{ece_noimg:.4f}" if ece_noimg is not None else "N/A"
    s_kl = f"{kl_val:.4f}" if kl_val is not None else "N/A"
    s_ent = f"{ent_val:.4f}" if ent_val is not None else "N/A"
    s_rho = f"{rho_val:.4f}" if rho_val is not None else "N/A"

    print(f"\n  📊 RESULTADOS: {run_name}")
    print(f"     ECE Geral (Erro de Calibração): {s_ece}")
    print(f"     ECE Com Imagem:                 {s_ece_img}")
    print(f"     ECE Sem Imagem:                 {s_ece_noimg}")
    print(f"     KL Divergence (vs Professor):   {s_kl}")
    print(f"     Entropia Média (Incerteza):     {s_ent}")
    print(f"     Correlação ρ(HR, HA):           {s_rho}")
    print(f"     Tempo do modelo:                {elapsed:.1f} minutos")
    print(f"     Salvo no Drive em:              {summary_path}\n", flush=True)

    del student
    torch.cuda.empty_cache()

# --- Resumo Consolidado em Tabela ---
print(f"\n{'═' * 70}")
print("📊 RESUMO GERAL DE CALIBRAÇÃO E INCERTEZA (SCIENCEQA)")
print(f"{'═' * 70}")
print(f"{'Experimento':32s} | {'ECE (Geral)':>11s} | {'ECE (Img)':>10s} | {'KL Div':>8s} | {'Entropia':>8s}")
print("-" * 75)

for name, s in summary_results.items():
    ece_g = f"{s.get('ece'):.4f}" if s.get('ece') is not None else "N/A"
    ece_i = f"{s.get('with_image_ece'):.4f}" if s.get('with_image_ece') is not None else "N/A"
    kl_v = f"{s.get('mean_kl'):.4f}" if s.get('mean_kl') is not None else "N/A"
    ent_v = f"{s.get('mean_entropy'):.4f}" if s.get('mean_entropy') is not None else "N/A"
    print(f"{name:32s} | {ece_g:>11s} | {ece_i:>10s} | {kl_v:>8s} | {ent_v:>8s}")

print(f"{'═' * 70}")
total_time = (time.time() - eval_start) / 60.0
print(f"Resultados salvos em: {summary_path}")
print(f"✓ CÉLULA F CONCLUÍDA em {total_time:.0f} minutos!")
print(f"{'═' * 70}\n")
