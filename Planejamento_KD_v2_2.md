# PLANEJAMENTO OPERACIONAL — v2.2

**Knowledge Distillation em LLMs · Artigo II (BRACIS 2026)**
**Data:** 23 de abril de 2026
**Status:** Pré-0 concluída · Fase 0 congelada · CoT gerado pelo teacher confirmado · Pronto para Fase 1.0 (piloto)

---

## Convenção de atribuição de tarefas

- 🧑 **HUMANO** — você executa: pesquisa adicional, decisões de escopo, validação de resultados, treino na sua infraestrutura, escrita do artigo final, conferência de figuras
- 🤖 **CLAUDE CODE** — agente de código executa: implementação, testes pytest, scripts de geração de dados, análise estatística, geração de figuras
- 💬 **CLAUDE (chat)** — eu, neste chat: planejamento, revisão de design, atualização de documentos, apoio em decisões metodológicas

---

## CHANGELOG v2.1 → v2.2 (ajuste ao código real)

| # | Alteração | Justificativa |
|---|---|---|
| 1 | Confirmado uso de **CoT gerado pelo teacher** (não resposta nativa GSM8K) | Consistência distribucional com hipótese H1; alinhamento com EOPD/TSD-KD |
| 2 | `src/data_gsm8k.py` existente já tem 3 regiões funcionando — **não reimplementar, adaptar** | Inventário do código real revelou tokenise_example completo |
| 3 | Detecção de `####` usa `text.find()` (robusto), não subsequência de tokens | Já implementado corretamente em `data_gsm8k.py` |
| 4 | `train_manual.py` já aceita `dataset: gsm8k` e `teacher_load_mode: bf16` | Sem modificação no runner |
| 5 | Tarefa 2.2 reformulada: **adaptar `data_gsm8k.py` para ler JSONL externo** (em vez de carregar HF direto) | Permite usar CoT do teacher no lugar de resposta nativa |
| 6 | Removida reimplementação da Collator; reutilizar `KDCollator` existente | Já suporta `region_ids` com padding `-1` |
| 7 | Adicionada decisão D6: filtrar exemplos sem separador `####` no output do teacher | Pré-0 A20 |
| 8 | Changelog v2.0→v2.1 preservado como histórico | Rastreabilidade |

## CHANGELOG v2.2 addendum (2026-04-24 · dry-run-driven)

Dry-run de 5 amostras no Colab (Qwen2.5-7B-Instruct bf16, greedy,
seed=42, indices fora de {0,1,2,3}) resultou em `separator_rate=0.0`,
`mean_total_len=912`, `p95_total_len=976`, `is_correct=0.0` — mas
inspeção qualitativa confirmou 4/5 respostas numericamente corretas
em `\boxed{N}`. Causa raiz: o 4-shot puro **não segura o formato
`####`** contra o prior RLHF do Qwen2.5-7B-Instruct em matemática
(consistente com Pré-0 [18]). Três fixes aplicados em
`scripts/pilot_cot_generation.py`:

| # | Fix | Alteração | Efeito esperado |
|---|---|---|---|
| F1 | System prompt reforçado | Instrução explícita "plain prose, no LaTeX, no `\boxed`, no markdown; end with `#### N`" | Alavanca principal para `separator_rate ≥ 0.95` |
| F2 | Fallback `\boxed{N}` | Regex secundário `\\boxed\{([^}]+)\}` alimenta `latent_accuracy_vs_gold`; **não gates** o go/no-go | Torna acurácia latente do teacher (A20) observável mesmo quando formato falha |
| F3 | Limpeza do few-shot | Remove `<<expr=result>>` dos 4 assistant turns via `re.sub(r"<<[^>]*>>", "", ans)`; prosa e `\n#### N` preservados | Remove artefato que empurra modelo para LaTeX; CoT dos exemplos continua sendo GSM8K nativo |

**Invariantes preservadas:** CoT gerado pelo teacher sobre as 8.788
questões de treino+teste continua sendo `P_teacher` (Fix 3 altera
apenas os 4 exemplos few-shot, não o alvo da destilação). `separator_found`
permanece gate estrito em `####` (load-bearing para
`tokenise_example` em `src/data_gsm8k.py`). Indices fixos `[0,1,2,3]`
e `numpy.random.RandomState(seed).choice` sobre o pool excluindo
few-shot permanecem inalterados.

**Próximo passo:** novo dry-run de 5; se `separator_rate ≥ 0.80` e
`latent_accuracy ≥ 0.80`, autorizar piloto de 100.

---

# PARTE I — ARTIGO I: CONCLUÍDO (SEMISH 2026)

> ℹ Sem alteração vs v2.1. Referência canônica imutável. Resultados do Artigo I são contexto, não premissas do Artigo II.

Limitações herdadas (estado no Artigo II):
- L1 (single seed) → **Resolvida** (3 seeds)
- L2 (professor 4-bit) → **Resolvida** (bf16)
- L3 (ECE adaptado de classificação) → **Persiste** — declarar
- L4 (Dolly genérico) → **Resolvida** (GSM8K)
- L5 (sem OOD independente) → **Persiste parcialmente**
- L6 (sem múltiplas seeds FKL) → **Resolvida**

---

# PARTE II — MOTIVAÇÃO CIENTÍFICA (SEM ALTERAÇÃO)

Preservada integralmente de v2.1. Pontos-chave:

- **Hipótese H1:** Sob épocas limitadas (3 épocas), RKL exerce *tail-focus* na distribuição do professor; em CoT onde ~20% dos tokens são "forking tokens" de alta entropia concentrados cedo no raciocínio [Wang et al. ACL 2025], o efeito deve ser assimétrico: **ρ_RKL = H_R/H_A < ρ_FKL**.
- **Hipótese H2:** Não-monotonicidade de temperatura na FKL persiste em CoT (análise complementar).
- **Posicionamento vs EOPD:** off-policy, comparação isolada (não híbrida), métricas segmentadas por fase, foco em calibração.
- **Contribuição:** primeira caracterização empírica controlada de ECE e H(t) segmentados por fase raciocínio/resposta em destilação CoT com FKL vs RKL isolados off-policy.

---

# PARTE III — FASE PRÉ-0 (CONCLUÍDA)

> ✔ Referência detalhada: `Pre0_Artigo_II_Consolidado.md` (33 artigos, alertas A11–A20).

---

# PARTE IV — FASE 0: ESCOPO CONGELADO (AJUSTADO AO CÓDIGO REAL)

## 4.1 — Pergunta central

Idêntica a v2.1: como a direção da KL (FKL vs RKL) afeta H(t) e ECE em destilação CoT off-policy do Qwen2.5-7B-Instruct para Qwen2.5-1.5B-Instruct em GSM8K, com análise segmentada por fase?

## 4.2 — Dataset (CONGELADO — ajustado para JSONL do teacher)

| Campo | Valor |
|---|---|
| Dataset principal | GSM8K (`openai/gsm8k`, config `main`) |
| Split treino | `train` — 7.473 exemplos |
| Split teste | `test` — 1.319 exemplos |
| **Target de treino** | **CoT gerado pelo teacher Qwen2.5-7B-Instruct bf16** (greedy, T=0.0) |
| Separador raciocínio→resposta | `\n#### <número>` (detectado por `text.find("####")` — padrão já em `data_gsm8k.py`) |
| Few-shot prompt | 4 exemplos do GSM8K train (formato nativo `#### N`) |
| Filtro pós-geração | Descartar exemplos do teacher onde `####` não aparece; threshold < 5% de descarte |
| Fonte de dados no treino | **JSONL externo**: `data/gsm8k_cot_qwen25_7b.jsonl` |
| Dataset secundário | MMLU (`cais/mmlu`, `all`, split `test`) — código já existe em `src/data_mmlu.py` |
| Tokenizer | Qwen2.5-1.5B-Instruct |
| Max sequence length | **768 tokens** (validar no piloto; fallback 1024) |
| Caching | JSONL gerado uma vez, reutilizado em todos os 21 runs |

## 4.3 — Modelos (CONGELADO)

| Campo | Valor | Observação |
|---|---|---|
| Professor | `Qwen/Qwen2.5-7B-Instruct` | `teacher_load_mode: bf16` (já suportado por `load_teacher`) |
| Estudante | `Qwen/Qwen2.5-1.5B-Instruct` | `student_dtype: bf16` (idêntico Artigo I) |
| Vocabulário | Idêntico | Check já em `train_manual.py` (linhas de verificação token-id) |
| Inicialização | Pré-treinada (Instruct) | Idêntico Artigo I |

## 4.4 — Funções de Perda (IMUTÁVEIS — reutilizar sem modificação)

Código existente em `src/losses_kd.py` é **imutável**:
- `compute_ce(shift_student_logits, shift_labels, valid_mask)` — já implementado
- `compute_kd_forward_kl(shift_teacher_logits, shift_student_logits, valid_mask, T)` — já implementado com T² explícito
- `compute_kd_reverse_kl(shift_student_logits, shift_teacher_logits, valid_mask)` — já implementado sem T², sem temperatura (MiniLLM Gu et al. 2024)
- `compute_total_loss(...)` — despacha por `kd_mode ∈ {ce_only, fkl, rkl}`

Todos com docstrings referenciando Hinton 2015 e Gu et al. 2024 — **convenção a seguir em todo código novo**.

## 4.5 — Hiperparâmetros (CONGELADOS)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Otimizador | AdamW | Artigo I (`torch.optim.AdamW` em `train_manual.py`) |
| Learning rate | 4e-5 | Artigo I, A19 |
| Scheduler | `get_cosine_schedule_with_warmup` | Já em `train_manual.py` |
| Warmup ratio | 0.1 | Artigo I |
| Weight decay | 0.05 | Artigo I |
| Épocas | 3 | Artigo I |
| Batch efetivo | 64 (batch 16 × grad_accum 4) | Artigo I |
| Max sequence length | **768** | A12 — validar no piloto |
| α | 0.5 fixo | Hinton 2015 |
| `max_grad_norm` | 1.0 | Artigo I |
| `teacher_load_mode` | **bf16** | Decisão do usuário |
| `student_dtype` | bf16 | Artigo I |

⚠ **Decisão D2 (potencial OOM):** se FKL bf16 com max_length=768 estourar memória, reduzir batch 16→8 e grad_accum 4→8 (mantém batch efetivo=64). `enable_gradient_checkpointing: true` também está disponível como opção no YAML.

## 4.6 — Métricas (REFINADAS — segmentação por fase)

### Notação
- `y_R` = tokens com `region_id == REGION_REASONING` (=1) e `labels != -100`
- `y_A` = tokens com `region_id == REGION_ANSWER` (=2) e `labels != -100`
- `|y| = |y_R| + |y_A|` (tokens com loss ativo)

### Métricas centrais

| Métrica | Definição | Observação |
|---|---|---|
| **H_R** | (1/\|y_R\|) Σ H(t) para t ∈ y_R, com H(t) = −Σ_v p_S(v\|...) log p_S(v\|...) | T=1 nos logits |
| **H_A** | (1/\|y_A\|) Σ H(t) para t ∈ y_A | T=1 |
| **H_total** | (1/\|y\|) Σ H(t) | Compatível com Artigo I |
| **H_R^norm** = H_R / log\|V\| | Entropia normalizada | Comparação cross-modelo |
| **H_A^norm** = H_A / log\|V\| | Idem | |
| **ρ = H_R / H_A** | **Métrica central da hipótese H1** | |
| **ECE_response** | 10 bins equal-width, só em `region_id == REGION_ANSWER` | A18 — fase de raciocínio excluída |
| **KL_R(T‖S)** | KL Teacher-Student média sobre y_R | T=1 |
| **KL_A(T‖S)** | KL Teacher-Student média sobre y_A | T=1 |
| **Acurácia GSM8K** | Exact match após `####` no output greedy do student | |
| **Taxa de falha de formato** | % de respostas sem `####` no output do student | Manter < 5% |

### Métricas auxiliares

- NLL e Perplexidade (totais e por fase)
- ECE em MMLU (sem segmentação — múltipla escolha)
- ROUGE-L (opcional, via `evaluate_generation.py` existente)

## 4.7 — Estatística (CONGELADA)

| Parâmetro | Valor |
|---|---|
| Seeds | 42, 123, 7 |
| Reportagem | Média ± DP sobre 3 seeds |
| Teste estatístico | t-test pareado bicaudal (CE vs FKL, CE vs RKL, FKL vs RKL); pareado por exemplo |
| Threshold | p < 0.05 + diferença > 1 DP |
| Total de runs | 21 = 3 CE (T=1) + 9 FKL (T∈{1,2,4} × 3 seeds) + 9 RKL (T∈{1,2,4} × 3 seeds) |
| Tempo estimado | ~50 min/run × 21 ≈ 17.5h |

## 4.8 — Restrições de escopo (HARD LIMITS)

Idênticas a v2.1: sem OOD independente, sem múltiplos estudantes, sem KD on-policy, sem comparação numérica direta com EOPD, sem solução híbrida.

---

# PARTE V — FASES DE EXECUÇÃO (AJUSTADAS AO CÓDIGO REAL)

## FASE 1.0 — Piloto de geração CoT (gate bloqueante)

**Objetivo:** validar formato, comprimento e acurácia em 100 exemplos antes de gerar 8.792.

**Critérios de aprovação (todos obrigatórios):**
- Aderência de `####` no output do teacher ≥ 95%
- Comprimento médio (prompt + CoT + answer) ≤ 750 tokens
- Percentil 95 do comprimento total ≤ 768
- Acurácia do teacher ≥ 80%

### Tarefas

| # | Tarefa | Responsável | Entregável |
|---|---|---|---|
| 1.0.1 | Implementar `scripts/pilot_cot_generation.py` (Nível B) | 🤖 Claude Code | Script |
| 1.0.2 | Definir prompt few-shot com 4 exemplos fixos do GSM8K train (seeds reproduzíveis) | 🤖 Claude Code | Constante `FEW_SHOT_EXAMPLES` no script |
| 1.0.3 | Executar piloto em 100 exemplos (Qwen2.5-7B bf16, greedy) | 🧑 Humano | `outputs/pilot/pilot_results.json` |
| 1.0.4 | Gerar `outputs/pilot/pilot_report.md` com decisão go/no-go | 🤖 Claude Code | Relatório |
| 1.0.5 | Validar relatório e autorizar Fase 1.1 | 🧑 Humano | Decisão |

### Especificação — `scripts/pilot_cot_generation.py` (Nível B)

```
Função principal: run_pilot(teacher_name, n_samples=100, seed=42,
                             max_new_tokens=512, output_dir="outputs/pilot")

Entradas:
  - teacher_name: "Qwen/Qwen2.5-7B-Instruct"
  - Carregar em bf16 via load_teacher() de src.train_manual (já implementada)

Algoritmo:
  1. Carregar GSM8K train via datasets.load_dataset("gsm8k", "main", split="train")
  2. Amostrar n_samples com numpy.random.RandomState(seed).choice
  3. Para cada exemplo:
     - Construir prompt: sistema + 4-shot + question atual
     - Aplicar chat template com add_generation_prompt=True
     - Gerar com model.generate(temperature=0.0, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
     - Extrair output (após prompt)
     - Detectar '####' via text.find() [mesmo padrão de data_gsm8k.py]
     - Extrair número por regex: r'####\s*(-?[\d,]+(?:\.\d+)?)'
     - Normalizar (remover vírgulas) e comparar com gold
  4. Salvar pilot_results.json com todos os campos

Few-shot prompt (escolher 4 exemplos DETERMINÍSTICOS do GSM8K train):
  - Usar indices fixos [0, 1, 2, 3] do split train — simples e reproduzível
  - Formato de cada exemplo:
      "Question: <q>\nAnswer: <full teacher_answer_with_####>"

Campos do pilot_results.json (por exemplo):
  {
    "idx": int, "question": str, "answer_gold": str,
    "prompt_text": str, "generated_text": str,
    "separator_found": bool, "extracted_answer": str | None,
    "is_correct": bool, "total_len_tokens": int,
    "prompt_len_tokens": int, "generation_len_tokens": int
  }

Convenção de citação na docstring:
  - Cobbe et al. 2021 (GSM8K, formato #### nativo)
  - Distilling Step-by-Step (Hsieh et al. 2023) — few-shot CoT prompting do teacher

Critério de teste (manual): rodar com n_samples=5 antes de rodar 100
                              para validar que o pipeline completa sem erro.
```

### Especificação — `outputs/pilot/pilot_report.md` (Nível A)

```
Gerar markdown com:
  ## Piloto de Geração CoT — Qwen2.5-7B-Instruct bf16
  ## Critérios de aprovação
  | Critério | Valor obtido | Threshold | Status |
  ## Distribuição de comprimentos (histograma textual 10 bins)
  ## Exemplos sem separador (até 5)
  ## Decisão: GO / NO-GO
  ## Justificativa (1 parágrafo)
  ## Se NO-GO, correção sugerida:
    - Refinar few-shot prompt / subir max_length para 1024 / filtrar por comprimento
```

---

## FASE 1.1 — Geração completa do CoT (após go do piloto)

| # | Tarefa | Responsável | Entregável |
|---|---|---|---|
| 1.1.1 | Implementar `scripts/generate_full_cot.py` (estender piloto) | 🤖 Claude Code | Script |
| 1.1.2 | Executar em GSM8K train (7.473) + test (1.319) | 🧑 Humano (~2–4h GPU) | `data/gsm8k_cot_qwen25_7b.jsonl` |
| 1.1.3 | Gerar `data/cot_stats.json` com estatísticas finais | 🤖 Claude Code | Stats |
| 1.1.4 | Conferir amostra qualitativa (10 exemplos) | 🧑 Humano | Aprovação |

### Especificação — `data/gsm8k_cot_qwen25_7b.jsonl` (Nível B)

Cada linha JSON:
```
{
  "split": "train" | "test",
  "idx": int,
  "question": str,              // pergunta GSM8K original
  "answer_gold": str,           // número gold extraído da answer GSM8K nativa
  "teacher_full_text": str,     // output completo do teacher (CoT + #### N)
  "extracted_answer": str | null,
  "is_teacher_correct": bool,
  "separator_found": bool,
  "total_len_tokens": int       // tamanho total estimado após tokenização
}
```

**Campos `rationale_len_tokens` e `answer_len_tokens` NÃO são calculados aqui.** A segmentação por fase é feita dinamicamente em `tokenise_example` (via `region_ids`) — mantém compatibilidade com o padrão do `data_gsm8k.py` atual.

**Filtro:** exemplos com `separator_found=False` são mantidos no JSONL (para rastreabilidade) mas **filtrados na função de carga**.

---

## FASE 2 — Adaptação do pipeline (ajustada)

> ℹ **Diretiva:** `src/losses_kd.py` é imutável. `src/data_gsm8k.py` existente é a base — **adaptar** para ler JSONL do teacher, não reimplementar. Branch: `cot-teacher-jsonl`.

### Tarefas

| # | Tarefa | Responsável | Entregável |
|---|---|---|---|
| 2.1 | Inventário do código existente | 🤖 Claude Code | Parágrafo no PR |
| 2.2 | Adaptar `src/data_gsm8k.py`: adicionar função `load_gsm8k_cot_jsonl()` + variante de `tokenise_example` que usa `teacher_full_text` como content do assistant | 🤖 Claude Code | Código + teste |
| 2.3 | Adicionar métricas em `src/analysis_metrics.py` (3 funções) | 🤖 Claude Code | Código + testes |
| 2.4 | Estender `src/evaluate_probabilistic.py` para reportar métricas por fase | 🤖 Claude Code | Código |
| 2.5 | Estender `src/evaluate_generation.py` para acurácia GSM8K (extração `####`) | 🤖 Claude Code | Código |
| 2.6 | 4 testes novos em `tests/test_cot_metrics.py` | 🤖 Claude Code | Testes passando |
| 2.7 | Sanity checks 12–15 em `src/sanity.py` | 🤖 Claude Code | Checks passando |
| 2.8 | Code review + merge | 🧑 Humano | Branch mergeada |

### 2.2 — Adaptação de `src/data_gsm8k.py` (Nível C — crítico)

Manter a função `tokenise_example` existente intocada para compatibilidade com resposta nativa. Adicionar:

```
def load_gsm8k_cot_jsonl(
    jsonl_path: str,
    split: str = "train",
    filter_no_separator: bool = True,
    filter_teacher_wrong: bool = False,   # D3 — default False, ativar se teacher accuracy < 80%
) -> List[Dict[str, Any]]:
    """Load teacher-generated CoT dataset from JSONL (Phase 1.1 output).

    Reference:
        Hsieh et al. 2023 (Distilling Step-by-Step) — teacher-generated
        rationales as supervision signal for student training.

    Returns: list of dicts with keys:
        question, teacher_full_text, answer_gold, is_teacher_correct,
        separator_found, idx, split.

    Filters:
        - By default removes examples where separator_found == False
        - Optionally removes examples where is_teacher_correct == False (D3)
    """

Algoritmo:
  1. Abrir JSONL, filtrar linhas por split
  2. Se filter_no_separator: descartar separator_found == False
  3. Se filter_teacher_wrong: descartar is_teacher_correct == False
  4. Reportar contagens (total, filtrados por cada critério)
  5. Retornar lista


def tokenise_example_cot(
    example: Dict[str, Any],   # linha do JSONL
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenise a GSM8K-CoT example using teacher-generated content.

    Idêntico ao tokenise_example() existente, exceto:
      - messages[2]["content"] = example["teacher_full_text"]  (não example["answer"])

    Mantém a mesma convenção de 3 regiões (REGION_PROMPT=0, REASONING=1, ANSWER=2)
    via detecção de '####' por text.find().

    Reference: MiniLLM (Gu et al. 2024) — off-policy KD consumes fixed
               teacher outputs; consistency with teacher distribution is
               critical for FKL/RKL comparison.
    """

Algoritmo: copiar tokenise_example existente, substituir example["answer"]
           por example["teacher_full_text"] na construção de messages.
           Resto do algoritmo (regions via text.find("####")) é idêntico.


def build_dataloader_cot(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    jsonl_path: str,
    split: str = "train",
    filter_teacher_wrong: bool = False,
    micro_overfit_n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """End-to-end: load JSONL → tokenise → DataLoader.

    Uses the existing KDCollator (supports region_ids natively).
    Reproducibility: uses torch.Generator(seed) and worker_init_fn
    (mesmo padrão do data_dolly.py).
    """
```

**Atualização em `train_manual.py`** (mínima): em `_build_dataloader`, quando `cfg["dataset"] == "gsm8k_cot"`, chamar `build_dataloader_cot(...)` com `jsonl_path=cfg["cot_data_path"]`. Manter `"gsm8k"` (resposta nativa) funcional para não quebrar runs anteriores.

### 2.3 — Métricas em `src/analysis_metrics.py` (Nível C)

As três funções já especificadas em v2.1 — **sem mudança em algoritmo**, apenas confirmando que consomem `region_ids` do `KDCollator` (já padded com `-1`):

```
def compute_entropy_by_phase(
    student_logits: torch.Tensor,   # [B, L, V]  (ANTES do shift causal)
    region_ids: torch.Tensor,       # [B, L]     (do KDCollator)
    valid_mask: torch.Tensor,       # [B, L] bool
    vocab_size: int,
    eps: float = 1e-8,
) -> Dict[str, float]

def compute_kl_by_phase(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    region_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]

def compute_ece_response_only(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    region_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]
```

**ATENÇÃO — shift causal:** todas as três funções devem aplicar `shift_for_causal_lm` internamente antes de calcular métricas, **seguindo o mesmo padrão de `compute_total_loss`**. Ou, alternativamente, aceitar que `evaluate_probabilistic.py` faça o shift e passe tensores já shifted — **decidir consistência com o que `analysis_metrics.py` já faz no Artigo I** (Claude Code deve ler o arquivo atual e seguir a convenção).

**Construção de `valid_mask`:** `valid_mask = attention_mask.bool() & (labels != -100)` (padrão do Artigo I em `losses_kd.py`).

### 2.6 — Testes pytest novos (Nível C)

```
tests/test_cot_metrics.py:

def test_entropy_by_phase_consistency():
    """H_total should equal the token-weighted average of H_R and H_A."""
    # Gerar logits sintéticos [B=2, L=10, V=100]
    # Construir region_ids com split conhecido (ex: 4 prompt, 3 reasoning, 3 answer)
    # valid_mask matching
    # assert abs(H_total * (n_R + n_A) - (n_R * H_R + n_A * H_A)) < 1e-4

def test_region_partition_sanity():
    """For a tokenised GSM8K-CoT example, all regions sum to seq_len."""
    # Mock tokenizer + example com '####' conhecido
    # Chamar tokenise_example_cot
    # assert count(region_ids == 0) + count(region_ids == 1) + count(region_ids == 2) == len(input_ids)

def test_ece_response_excludes_reasoning():
    """ECE_response uses only REGION_ANSWER tokens."""
    # Logits sintéticos com confidences previsíveis em cada fase
    # ECE_response deve bater com cálculo manual restrito a região 2
    # (verificar que mudanças em região 1 não afetam ECE_response)

def test_jsonl_loading_filters():
    """load_gsm8k_cot_jsonl filters separator_found==False correctly."""
    # Criar JSONL temporário com 10 entradas, 2 com separator_found=False
    # Carregar com filter_no_separator=True → retorna 8
    # Carregar com filter_no_separator=False → retorna 10
```

### 2.7 — Sanity checks novos em `src/sanity.py` (Nível B)

```
check_12_jsonl_exists_and_valid:
    Verifica data/gsm8k_cot_qwen25_7b.jsonl existe, tem campos esperados,
    reporta contagens por split e filtros.

check_13_region_ids_in_batch:
    Carrega 1 batch do dataloader CoT, verifica presence de region_ids
    com valores em {-1, 0, 1, 2} e que cada exemplo tem todas as 3 regiões.

check_14_separator_token_stability:
    Tokeniza '####' isoladamente no tokenizer Qwen2.5-1.5B, registra o
    número de tokens (esperado: 1). Alerta se > 1.

check_15_phase_metrics_finite:
    Roda compute_entropy_by_phase em um batch real, verifica H_R, H_A finitos,
    rho em range razoável [0.1, 5.0].
```

---

## FASE 3 — Revisão matemática (0.5 dia)

| # | Tarefa | Responsável |
|---|---|---|
| 3.1 | Validar fórmulas H(t), ECE, KL por fase em documento separado | 💬 Claude (chat) |
| 3.2 | Conferência final antes de runs | 🧑 Humano |
| 3.3 | Confirmar docstrings do código batem com fórmulas | 🤖 Claude Code |

---

## FASE 4 — Replicação Controlada (21 runs)

### Configs

| # | Tarefa | Responsável |
|---|---|---|
| 4.0.1 | Gerar 21 configs YAML `configs/gsm8k_cot_*.yaml` | 🤖 Claude Code |
| 4.0.2 | Validar 1 config de cada condição | 🧑 Humano |

**Naming:** `gsm8k_cot_<kd_mode>_T<temp>_seed<seed>.yaml`

**Campos novos no YAML:**
```yaml
dataset: gsm8k_cot                              # NOVO — aciona build_dataloader_cot
cot_data_path: data/gsm8k_cot_qwen25_7b.jsonl   # NOVO
max_length: 768                                  # mudança vs Dolly
teacher_load_mode: bf16                          # mudança vs Artigo I
student_dtype: bf16
filter_teacher_wrong: false                      # D3, default false
```

### Execução

Ordem idêntica a v2.1: CE → FKL T=1 → T=2 → T=4 → RKL T=1 → T=2 → T=4, sempre 3 seeds (42, 123, 7).

### Monitoramento durante cada run

| Item | Threshold | Responsável |
|---|---|---|
| Loss total decrescente | obrigatório | 🧑 Humano (logs) |
| KL(T‖S) decrescendo (FKL/RKL) | obrigatório | 🤖 Claude Code (script check) |
| Taxa de falha `####` no output do student | < 5% | 🤖 Claude Code |
| Seed divergente | investigar | 🧑 Humano |

### Pós-treino por run

| # | Tarefa | Responsável |
|---|---|---|
| 4.1 | `evaluate_probabilistic.py` em `checkpoints/<run>/final` | 🤖 Claude Code |
| 4.2 | `evaluate_generation.py` (acurácia GSM8K + opcional ROUGE) | 🤖 Claude Code |
| 4.3 | Salvar consolidado em `results/gsm8k_cot_<run_name>.json` | 🤖 Claude Code |

---

## FASE 5 — Análise multi-métrica

| # | Tarefa | Responsável | Entregável |
|---|---|---|---|
| 5.1 | Agregar 21 runs (média ± DP) | 🤖 Claude Code | `results/aggregated.parquet` |
| 5.2 | t-tests pareados | 🤖 Claude Code | `results/significance.json` |
| 5.3 | Tabela 3 do artigo | 🤖 Claude Code | `results/table_3.tex` |
| 5.4 | Figura: ρ por condição com error bars | 🤖 Claude Code | `figures/rho_by_condition.pdf` |
| 5.5 | Figura: H_R vs H_A scatter | 🤖 Claude Code | `figures/hr_vs_ha.pdf` |
| 5.6 | Figura: H(t) por posição com marcador `####` | 🤖 Claude Code | `figures/ht_position.pdf` |
| 5.7 | Validar H1 (ρ_RKL < ρ_FKL?) | 🧑 Humano | Análise textual |
| 5.8 | Comparação qualitativa com Artigo I | 🧑 Humano | Discussão |

> ℹ Se H1 for rejeitada, o artigo vira "caracterização não-óbvia que contraria a expectativa baseada em EOPD". Não reescrever hipótese a posteriori.

---

## FASE 6 — Micro-exploração (opcional)

Idêntica a v2.1. Candidatos pré-aprovados:
- (a) ECE por terço da fase de raciocínio — justificado por [Zhao 2603.18940]
- (b) Análise de forking tokens no student — justificado por [Wang et al. ACL 2025]

---

## FASE 7 — Escrita do artigo

Idêntica a v2.1. Checklist de Limitações inclui L3, L5 (persistentes), A11–A20 (novos).

---

# PARTE VI — REGRAS GERAIS E CRONOGRAMA

## 6.1 — Regras gerais

- 🧑 Não adicionar nova variável sem justificativa teórica da Pré-0
- 🧑 Não expandir escopo após congelamento
- 🤖 Sempre registrar seeds; reportar média ± DP
- 🤖 Toda função nova deve ter teste numérico antes do treino
- 🤖 Toda função nova deve ter docstring com **referência bibliográfica + fórmula** (convenção herdada de `losses_kd.py`)
- 💬 Decisões em execução → CHANGELOG antes de prosseguir
- 🧑 Comparação qualitativa com Artigo I é obrigatória na discussão
- 🤖 `src/losses_kd.py` é **imutável**
- 🤖 `src/data_gsm8k.py` existente é base — **adaptar, não reimplementar**

## 6.2 — Cronograma estimado (prazo: 4 de maio de 2026)

| Fase | Atividade | Duração | Responsável | Data alvo |
|---|---|---|---|---|
| Pré-0 | Buscas | ✔ Concluído | 💬 | 21/abr |
| 0 | v2.2 + congelamento | ✔ Hoje | 💬 + 🧑 | 23/abr |
| 1.0 | Piloto CoT | 0.5 dia | 🤖 + 🧑 | 24/abr |
| 1.1 | Geração completa | 1 dia | 🧑 (GPU) | 25/abr |
| 2 | Adaptação pipeline | 1–2 dias | 🤖 | 26–27/abr |
| 3 | Revisão matemática | 0.5 dia | 💬 + 🧑 | 27/abr |
| 4 | 21 runs | 2–3 dias | 🧑 (GPU) | 28–30/abr |
| 5 | Análise | 1–2 dias | 🤖 + 🧑 | 30/abr–1/mai |
| 7 | Escrita | 3 dias | 🧑 | 1–4/mai |
| | **Submissão BRACIS** | | 🧑 | **4/mai** |

⚠ **Caminho crítico:** Piloto → Geração → Adaptação → 21 runs. Análise e escrita paralelizáveis nos últimos dias.

## 6.3 — Pontos de decisão pendentes

| # | Decisão | Quando | Responsável | Critério |
|---|---|---|---|---|
| D1 | Max length final (768 vs 1024) | Após Fase 1.0 | 🧑 + 💬 | Percentil 95 do piloto |
| D2 | Reduzir batch para 8+grad_accum 8 | Início Fase 4 | 🧑 | OOM em FKL bf16 |
| D3 | Filtrar teacher-wrong do JSONL | Após Fase 1.1 | 🧑 + 💬 | Se acurácia teacher < 80% |
| D4 | Aprovar Fase 6 (micro-exploração) | Após Fase 5 | 🧑 | Tempo restante |
| D5 | Reescrita parcial de H1 se rejeitada | Fase 7 | 🧑 + 💬 | Análise estatística |
| D6 | Ativar `enable_gradient_checkpointing` | Início Fase 4 | 🧑 | OOM persistente |

---

**FIM — v2.2 · CoT do teacher · código real como base · pronto para Fase 1.0**
