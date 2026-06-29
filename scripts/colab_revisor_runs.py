"""
Pipeline de resposta aos revisores — GSM8K-CoT KD (FKL/RKL/CE).

Use no Google Colab. Cada bloco `# %%` é uma célula (Colab/Jupyter reconhecem o
marcador). Diferente de um notebook, aqui os comandos de shell são chamados via
subprocess, então o arquivo é Python VÁLIDO e executável — não há magics
(`!`, `%cd`) para descomentar.

Cobertura, do mais barato ao mais caro:
  Célula 2  W2 + Review 3   -> SEM treino (análise top-2 + student base)
  Célula 3  α=0 (Review 1/2) e reweighting (W3) -> 3 treinos
  Célula 4  Seeds 123/7 de CE e FKL T4 (W1 Critical) -> 4 treinos

Pré-requisitos já confirmados:
  - Branch git: kd-ablations-reweighting (contém a correção do run.py que evita
    sobrescrever os runs principais no Drive).
  - Checkpoint FKL T4 seed42 já existe no Drive COM pesos
    (SLM_results/fkl_T4_seed42/checkpoints/final/model.safetensors, ~3 GB).
  - data/gsm8k_cot_qwen25_7b.jsonl vem junto no clone (versionado no repo).

Como rodar no Colab:
  - Recomendado: abrir como notebook (File > Open) e executar célula a célula.
  - Ou, numa célula: `%run scripts/colab_revisor_runs.py` (roda tudo de uma vez).

IMPORTANTE: os treinos das células 3 e 4 carregam teacher 7B + student 1.5B.
Em GPUs menores (T4 16 GB) ajuste batch_size/teacher_load_mode nos configs.
"""

import os
import subprocess
import sys

# Caminho do projeto e do Drive. Ajuste DRIVE se o seu SLM_results for outro.
REPO_DIR = "/content/SLMs-CoT"
REPO_URL = "https://github.com/Mavitu56/SLMs-CoT.git"
BRANCH = "kd-ablations-reweighting"
DRIVE = "/content/drive/MyDrive/SLM_results"


def run(cmd: list[str], cwd: str | None = None) -> None:
    """Roda um comando, ecoando-o, e aborta se falhar (mostra a saída)."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


# %%
# =====================================================================
# Célula 1 — Setup: clone do branch, dependências e Google Drive
# =====================================================================
def cell1_setup() -> None:
    # GPU disponível?
    subprocess.run(["nvidia-smi"], check=False)

    # Clone limpo do branch certo (traz o .jsonl de dados junto).
    if os.path.isdir(REPO_DIR):
        subprocess.run(["rm", "-rf", REPO_DIR], check=True)
    run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])

    # Dependências.
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # Google Drive.
    from google.colab import drive
    drive.mount("/content/drive")
    assert os.path.isdir(DRIVE), f"SLM_results não encontrado em {DRIVE} — ajuste DRIVE."
    print("Drive OK:", DRIVE)

    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)


# %%
# =====================================================================
# Célula 2 — W2 + Review 3  (SEM TREINO — faça isto primeiro)
# =====================================================================
def cell2_analysis_no_training() -> None:
    os.chdir(REPO_DIR)

    # Review 3: student BASE (não treinado) — baixa Qwen/Qwen2.5-1.5B-Instruct.
    run([
        sys.executable, "scripts/analyze_answer_competition.py",
        "--config", "configs/gsm8k/gsm8k_cot_ce_only_T1_seed42.yaml",
        "--checkpoint", "Qwen/Qwen2.5-1.5B-Instruct",
        "--label", "BASE",
        "--output", "results/w2_base.json",
    ])

    # W2: FKL T4 treinado — usa o checkpoint do DRIVE (pesos confirmados).
    fkl_ckpt = os.path.join(DRIVE, "fkl_T4_seed42", "checkpoints", "final")
    run([
        sys.executable, "scripts/analyze_answer_competition.py",
        "--config", "configs/gsm8k/gsm8k_cot_fkl_T4_seed42.yaml",
        "--checkpoint", fkl_ckpt,
        "--label", "FKL_T4",
        "--output", "results/w2_fkl_t4.json",
    ])


# %%
# =====================================================================
# Célula 3 — Ablações α=0 (Review 1/2) e reweighting 17.5x (W3)
# =====================================================================
# run.py corrigido: cada config grava em pasta PRÓPRIA no Drive
# (fkl_T4_alpha0_seed42 / rkl_T1_alpha0_seed42 / fkl_T4_rw_seed42); nenhum
# sobrescreve os runs principais (fkl_T4_seed42, rkl_T1_seed42).
ABLATION_CONFIGS = [
    "gsm8k_cot_fkl_T4_alpha0_seed42",  # FKL puro (α=0)
    "gsm8k_cot_rkl_T1_alpha0_seed42",  # RKL puro (α=0)
    "gsm8k_cot_fkl_T4_rw_seed42",      # reweighting da resposta 17.5x
]


def cell3_ablations() -> None:
    os.chdir(REPO_DIR)
    for cfg in ABLATION_CONFIGS:
        print(f"\n===== Treinando {cfg} =====")
        run([
            sys.executable, "scripts/run.py",
            "--config", f"configs/gsm8k/{cfg}.yaml",
            "--drive-root", DRIVE,
        ])


# %%
# =====================================================================
# Célula 4 — Seeds extras 123 e 7 de CE e FKL T4  (W1 Critical: média ± DP)
# =====================================================================
SEED_CONFIGS = [
    "gsm8k_cot_ce_only_T1_seed123",
    "gsm8k_cot_ce_only_T1_seed7",
    "gsm8k_cot_fkl_T4_seed123",
    "gsm8k_cot_fkl_T4_seed7",
]


def cell4_extra_seeds() -> None:
    os.chdir(REPO_DIR)
    for cfg in SEED_CONFIGS:
        print(f"\n===== Treinando {cfg} =====")
        run([
            sys.executable, "scripts/run.py",
            "--config", f"configs/gsm8k/{cfg}.yaml",
            "--drive-root", DRIVE,
        ])


# %%
# =====================================================================
# Célula 5 — Acurácia GSM8K + ECE + KL (run_analysis.py --run-gsm8k)
# =====================================================================
# O run.py (células 3-4) só treina e plota loss. A acurácia GSM8K e as
# métricas probabilísticas (ECE, KL(T||S), entropia) vêm de run_analysis.py,
# um passo SEPARADO sobre os checkpoints. Aqui rodamos --run-gsm8k para os 7
# runs novos E para os baselines seed42 (para a Tabela 2 / W1 ter média±DP na
# mesma métrica). O resultado vai para {pasta}/analysis/analysis_results.json.
#
# Mapeamento: label -> (pasta_no_drive, stem_do_config). O config fornece
# teacher/dataset; o checkpoint vem da pasta. Pasta != config nos baselines
# antigos (pasta fkl_T4_seed42, config gsm8k_cot_fkl_T4_seed42), por isso o
# mapeamento é explícito.
ANALYSIS_RUNS = {
    # --- Ablações (Review 1/2, W3) ---
    "FKL_T4_alpha0":  ("fkl_T4_alpha0_seed42", "gsm8k_cot_fkl_T4_alpha0_seed42"),
    "RKL_T1_alpha0":  ("rkl_T1_alpha0_seed42", "gsm8k_cot_rkl_T1_alpha0_seed42"),
    "FKL_T4_rw":      ("fkl_T4_rw_seed42",     "gsm8k_cot_fkl_T4_rw_seed42"),
    # --- Seeds extras (W1) ---
    "CE_seed123":     ("ce_only_T1_seed123",   "gsm8k_cot_ce_only_T1_seed123"),
    "CE_seed7":       ("ce_only_T1_seed7",     "gsm8k_cot_ce_only_T1_seed7"),
    "FKL_T4_seed123": ("fkl_T4_seed123",       "gsm8k_cot_fkl_T4_seed123"),
    "FKL_T4_seed7":   ("fkl_T4_seed7",         "gsm8k_cot_fkl_T4_seed7"),
    # --- Baselines seed42 (sem acurácia ainda; necessários p/ comparação) ---
    "CE_seed42":      ("ce_only_T1_seed42",    "gsm8k_cot_ce_only_T1_seed42"),
    "FKL_T1_seed42":  ("fkl_T1_seed42",        "gsm8k_cot_fkl_T1_seed42"),
    "FKL_T2_seed42":  ("fkl_T2_seed42",        "gsm8k_cot_fkl_T2_seed42"),
    "FKL_T4_seed42":  ("fkl_T4_seed42",        "gsm8k_cot_fkl_T4_seed42"),
    "RKL_T1_seed42":  ("rkl_T1_seed42",        "gsm8k_cot_rkl_T1_seed42"),
    "RKL_T2_seed42":  ("rkl_T2_seed42",        "gsm8k_cot_rkl_T2_seed42"),
    "RKL_T4_seed42":  ("rkl_T4_seed42",        "gsm8k_cot_rkl_T4_seed42"),
}


def cell5_analysis(
    only=None,
    max_batches=None,
    gsm8k_batch_size=64,
    no_teacher=False,
) -> None:
    """Roda run_analysis.py --run-gsm8k nos checkpoints do Drive.

    only:             lista de labels para um subconjunto (ex.: ["FKL_T4_seed123"]).
                      Use para cronometrar 1 run antes de disparar todos.
    max_batches:      limita batches da avaliação (smoke test). None = split inteiro.
    gsm8k_batch_size: batch da GERAÇÃO greedy (gargalo). Na A100 80GB o student
                      1.5B é pequeno, então 64 acelera muito vs. o default 8 do
                      config. Reduza se houver OOM (improvável na A100).
    no_teacher:       se True, pula o teacher 7B e NÃO calcula ECE/KL(T||S) —
                      só a acurácia GSM8K. Mais rápido; use se já tem o ECE ou
                      vai rodá-lo depois.

    Tempo estimado na A100 80GB (cenário realista, gsm8k_batch_size=64):
      ~5-6 min/checkpoint com teacher;  ~3-4 min/checkpoint com no_teacher=True.
    """
    os.chdir(REPO_DIR)
    items = ANALYSIS_RUNS.items() if only is None else [
        (k, ANALYSIS_RUNS[k]) for k in only
    ]
    for label, (folder, cfg_stem) in items:
        ckpt = os.path.join(DRIVE, folder, "checkpoints", "final")
        out_dir = os.path.join(DRIVE, folder, "analysis")
        cfg_path = f"configs/gsm8k/{cfg_stem}.yaml"
        if not os.path.isdir(ckpt):
            print(f"[skip] {label}: checkpoint não encontrado em {ckpt}")
            continue
        print(f"\n===== Analisando {label} ({folder}) =====")
        cmd = [
            sys.executable, "scripts/run_analysis.py",
            "--config", cfg_path,
            "--checkpoint", ckpt,
            "--label", label,
            "--output-dir", out_dir,
            "--run-gsm8k",
            "--gsm8k-batch-size", str(gsm8k_batch_size),
        ]
        if no_teacher:
            cmd += ["--no-teacher"]
        if max_batches is not None:
            cmd += ["--max-batches", str(max_batches)]
        run(cmd)


# %%
# =====================================================================
# Execução direta: roda o pipeline inteiro na ordem barato -> caro.
# (No Colab, prefira chamar as funções célula a célula.)
# =====================================================================
if __name__ == "__main__":
    cell1_setup()
    cell2_analysis_no_training()
    cell3_ablations()
    cell4_extra_seeds()
    cell5_analysis()
