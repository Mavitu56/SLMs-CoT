# SLMs-CoT: Knowledge Distillation em LLMs Autoregressivos

> **Status:** Experimento em andamento · Artigo II (BRACIS 2026)

Repositório de experimento científico que investiga como a **direção da divergência KL** em Knowledge Distillation (KD) afeta o comportamento probabilístico interno de modelos de linguagem pequenos. Especificamente, comparamos três condições de treino — CE baseline, Forward KL e Reverse KL — e medimos entropia token-level, calibração (ECE) e divergência Teacher–Student sobre sequências de raciocínio.

---

## Objetivo

Responder à seguinte pergunta de pesquisa:

> *Como o objetivo de distilação (FKL vs RKL) afeta a entropia token-level e a calibração do student em diferentes regiões de uma sequência de raciocínio (prompt / raciocínio / resposta)?*

As métricas centrais são:
- **H(t)** — Entropia token-level do student (T=1, logits crus)
- **ECE** — Expected Calibration Error com 10 bins equal-width
- **KL(T‖S)** — Divergência Forward do teacher para o student (avaliação, T=1)

---

## Modelos e Dados

| Papel | Modelo |
|---|---|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` (congelado, 4-bit por padrão) |
| Student | `Qwen/Qwen2.5-1.5B-Instruct` (bf16) |

| Dataset | Uso |
|---|---|
| `databricks/databricks-dolly-15k` | Treino principal (Artigo I) |
| `openai/gsm8k` | Treino com CoT teacher (Artigo II, em andamento) |
| `cais/mmlu` | Avaliação secundária (exact-match) |

---

## Estrutura do Repositório

```
configs/
  dolly/              # 21 configs YAML (3 CE + 9 FKL + 9 RKL) para Dolly

scripts/
  run.py              # Entrada de treino: carrega config → sanity → train → plots
  run_analysis.py     # Avaliação probabilística de checkpoints salvos
  pilot_cot_generation.py   # Fase 1.0: geração piloto de CoT (100 exemplos, COMPLETO)
  generate_full_cot.py      # Fase 1.1: geração completa do JSONL GSM8K (em andamento)

src/
  data/
    data_gsm8k.py     # GSM8K: tokenização com region_ids (prompt/raciocínio/resposta)
    data_dolly.py     # Dolly-15k: masking de prompt, collator dinâmico
    data_mmlu.py      # MMLU: MCQ, sem shuffle, apenas avaliação
  losses/
    losses_kd.py      # CE, FKL (Hinton 2015), RKL (MiniLLM) — IMUTÁVEL
  training/
    train_manual.py   # Loop PyTorch puro: AdamW + cosine scheduler + gradient accum
    sanity.py         # 11 checks pré-treino (teacher frozen, KL≈0, masking, etc.)
    utils_seed.py     # set_seed() determinístico (Python/NumPy/PyTorch/CUDA)
  evaluation/
    analysis_metrics.py         # Entropia, ECE (10-bin equal-width), KL por posição, NLL
    evaluate_probabilistic.py   # Pass de avaliação completo com breakdown por região
    evaluate_generation.py      # ROUGE-L (Dolly) e exact-match (MMLU)
  visualization/
    plotting_utils.py   # Gráficos comparativos de métricas e curvas por posição

tests/
  test_losses_numeric.py   # 6 testes analíticos em tensores sintéticos (todos passando)

data/                # Gerado localmente — não versionado
  gsm8k_cot_qwen25_7b.jsonl   # JSONL com CoT do teacher (Fase 1.1, em geração)
  cot_stats.json               # Estatísticas agregadas por split

outputs/
  pilot/
    pilot_report.md      # Relatório da Fase 1.0 com critérios de gate e decisão GO
    pilot_results.json   # Resultados brutos do piloto (100 exemplos)
```

---

## Condições Experimentais

| Condição | `kd_mode` | Fórmula | Temperatura |
|---|---|---|---|
| CE Baseline | `ce_only` | `L = L_CE` | — |
| Forward KL | `fkl` | `L = α·L_CE + (1−α)·T²·L_FKL` | T ∈ {1, 2, 4} |
| Reverse KL | `rkl` | `L = α·L_CE + (1−α)·L_RKL` | — (sem T²) |

Parâmetros congelados: `α=0.5`, `lr=2e-5`, `scheduler=cosine`, `warmup_ratio=0.1`, `weight_decay=0.05`, `num_epochs=3`, `batch_size_efetivo=8`, `max_length=512` (Dolly) / `1024` (GSM8K-CoT), `seeds={42, 123, 7}`.

Total de runs (Dolly): **21** (3 CE + 9 FKL + 9 RKL).

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

Requer Python 3.10+ e GPU com CUDA.

---

## Como Usar

### Treino

```bash
# Rodar um experimento
python scripts/run.py --config configs/dolly/dolly_fkl_T2_seed42.yaml

# Salvar em pasta específica
python scripts/run.py --config configs/dolly/dolly_fkl_T2_seed42.yaml \
    --output-dir results/fkl_T2_seed42

# Usar Google Drive (Colab)
python scripts/run.py --config configs/dolly/dolly_fkl_T2_seed42.yaml \
    --drive-root /content/drive/MyDrive/SLM_results
```

### Avaliação Probabilística

```bash
# Checkpoint único
python scripts/run_analysis.py \
    --config configs/dolly/dolly_fkl_T2_seed42.yaml \
    --checkpoint checkpoints/fkl_T2_seed42/final \
    --label FKL

# Comparação de múltiplos checkpoints
python scripts/run_analysis.py \
    --config configs/dolly/dolly_fkl_T2_seed42.yaml \
    --checkpoints \
        CE=checkpoints/ce_T1_seed42/final \
        FKL=checkpoints/fkl_T2_seed42/final \
        RKL=checkpoints/rkl_T2_seed42/final \
    --output-dir analysis_output

# Com métricas de geração (mais lento)
python scripts/run_analysis.py \
    --config configs/dolly/dolly_fkl_T2_seed42.yaml \
    --checkpoint checkpoints/fkl_T2_seed42/final \
    --label FKL --run-rouge --run-mmlu
```

### Testes

```bash
pytest tests/ -v
pytest tests/test_losses_numeric.py -v   # só os testes de losses
```

---

## Estado Atual

| Fase | Descrição | Status |
|---|---|---|
| Artigo I — Dolly KD | 21 configs criadas, treino com CE/FKL/RKL | Infra pronta |
| Fase 1.0 — Piloto CoT | Geração de 100 exemplos com Qwen2.5-7B, validação de gate | **COMPLETO (GO)** |
| Fase 1.1 — JSONL completo | Geração de 8.788 exemplos para `data/gsm8k_cot_qwen25_7b.jsonl` | Em andamento |
| Fase 2 — Adaptação de dados | `data_gsm8k.py` para consumir JSONL + novas métricas CoT | Pendente |
| Fase 4 — Runs GSM8K-CoT | 21 runs com o JSONL gerado | Pendente |
| Fase 5 — Análise | Correlação H_R vs acurácia, comparação FKL/RKL por região | Pendente |

Branch ativa: `cot-teacher-jsonl`.

---

## Referências

- Hinton et al. (2015) — *Distilling the Knowledge in a Neural Network* · [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Gu et al. (2024) — *MiniLLM: Knowledge Distillation of Large Language Models* · [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- Ko et al. (2024) — *DistiLLM: Towards Streamlined Distillation for Large Language Models* · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)
- Cobbe et al. (2021) — *Training Verifiers to Solve Math Word Problems* (GSM8K) · [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Hsieh et al. (2023) — *Distilling Step-by-Step* · [arXiv:2212.09561](https://arxiv.org/abs/2212.09561)

A revisão bibliográfica completa (33 referências) está em `docs/`.
