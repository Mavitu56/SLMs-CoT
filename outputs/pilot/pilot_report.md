# Piloto de Geração CoT — Qwen2.5-7B-Instruct bf16

**Fase:** 1.0 (piloto bloqueante) — Planejamento KD v2.2
**Data:** 2026-04-24
**Script:** `scripts/pilot_cot_generation.py` (commits `9a1d21b`, `f119fd0`)
**Branch:** `cot-teacher-jsonl`
**Resultado:** **GO** com ajuste de D1 (`max_length = 1024`)

---

## 1. Configuração

| Campo | Valor |
|---|---|
| `teacher_name` | `Qwen/Qwen2.5-7B-Instruct` |
| `teacher_load_mode` | `bf16` |
| `n_samples` | 100 |
| `seed` | 42 |
| `max_new_tokens` | 512 |
| `few_shot_indices` | `[0, 1, 2, 3]` (GSM8K train, fixos) |
| Amostragem | `numpy.random.RandomState(42).choice` sobre pool `GSM8K-train \ {0,1,2,3}` |
| Decoding | greedy (`do_sample=False`, `num_beams=1`, `temperature=0.0`) |
| `sep_token_count_qwen25` | 1 (`####` é 1 token no tokenizer Qwen2.5) |
| Fixes ativos | F1 (system prompt reforçado) · F2 (fallback `\boxed{}` informativo) · F3 (limpeza `<<expr=result>>` no few-shot) |

---

## 2. Critérios de aprovação

| Critério | Valor obtido | Threshold | Status |
|---|---|---|---|
| `separator_rate`                        | **1.000**  | ≥ 0.95 | **PASS** |
| `mean_total_len_tokens`                 | **721.2**  | ≤ 750  | **PASS** |
| `p95_total_len_tokens`                  | **832.4**  | ≤ 768  | **FAIL** (aciona D1) |
| `teacher_accuracy_vs_gold` (strict `####`) | **0.920** | ≥ 0.80 | **PASS** |
| `latent_accuracy_vs_gold` (`####` \| `\boxed`) | 0.920 | — (informativo) | — |
| `n_boxed_fallback_only`                 | 0          | — (informativo) | — |
| `n_truncated_by_max_new_tokens`         | 0          | — | — |

Três dos quatro gates passam com folga. O gate `p95 ≤ 768` é o único
fora, e foi desenhado exatamente para acionar a **decisão D1**
(`max_length` final, v2.2 §4.5 e §6.3) — não representa falha
substantiva do pipeline.

---

## 3. Distribuição de comprimentos (prompt + geração)

Histograma ASCII, 10 bins equal-width em `[600, 1000)`:

```
  [ 600,  640):   5  #####
  [ 640,  680):  22  ######################
  [ 680,  720):  28  ############################
  [ 720,  760):  24  ########################
  [ 760,  800):  10  ##########
  [ 800,  840):   6  ######
  [ 840,  880):   2  ##
  [ 880,  920):   1  #
  [ 920,  960):   2  ##
  [ 960, 1000):   0  (nenhuma amostra >= 960)
```

Percentis observados (`total_len_tokens`):

| estatística | valor |
|---|---|
| min   | 627 |
| p50   | 715 |
| p75   | 745 |
| p90   | 802 |
| p95   | 832 |
| p98   | 889 |
| p99   | 929 |
| max   | 956 |

Contagens contra thresholds candidatos:

| threshold | amostras acima | % |
|---|---|---|
| 768   | 16/100 | 16.0% |
| 800   | 11/100 | 11.0% |
| 900   |  2/100 |  2.0% |
| 1024  |  0/100 |  0.0% |

Geração isolada (`generation_len_tokens`, sem prompt): mean=115.8,
p95=208, max=306. Prompt é constante ~605 tokens (4-shot + system).
Ninguém chega perto de `max_new_tokens=512`, portanto **não há
truncamento mascarado**.

---

## 4. Exemplos sem separador `####`

**Zero.** 100/100 amostras terminam com `#### N`. Fix 1 (system prompt
reforçado) e Fix 3 (limpeza do few-shot) resolveram completamente o
problema de formato identificado no dry-run inicial (onde
`separator_rate` era 0.000 devido a emissão de `\boxed{N}` em LaTeX).

---

## 5. Erros substantivos do teacher (com `####` válido, resposta errada)

8/100 = 8% — consistente com `teacher_accuracy = 0.92`.

| idx | gold | pred | gen_len | natureza |
|---|---|---|---|---|
| 3324 | 35   | 14   | 121 | erro de interpretação |
| 4108 | 6    | 18   | 175 | erro aritmético |
| 5599 | 184  | 264  | 69  | erro de interpretação |
| 5834 | 32   | 71   | 69  | erro de interpretação |
| 6629 | 68   | 66   | 154 | erro aritmético (já errou no dry-run com LaTeX) |
| 6724 | 146  | 70   | 114 | erro de interpretação |
| 6755 | 2    | 5    | 306 | problema mais longo do piloto; teacher errou |
| 6917 | 4    | 3.98 | 115 | **arredondamento** — teacher resolveu o cálculo correto |

Exceto idx 6917 (que é tecnicamente um acerto com política `is_correct`
mais branda), são erros de raciocínio genuínos do `Qwen2.5-7B-Instruct`
não-math-específico. A acurácia observada (0.92) é coerente com a
estimativa de Pré-0 §A20 (80–88% para `Qwen2.5-7B-Instruct`) e supera
essa faixa — provavelmente efeito do system prompt de F1 que reforça
"plain prose, step by step".

Decisão **D3 (filtrar teacher-wrong)**: *não ativar*. 8% de descarte
não compensa a perda de diversidade no `P_teacher`, e o threshold do
v2.2 §6.3 D3 (ativar "se acurácia teacher < 80%") não é atingido.

---

## 6. As 3 amostras mais longas

| idx | total | gen | correct | pergunta (truncada) |
|---|---|---|---|---|
| 6755 | 956 | 306 | ✗ | Alexander draws 9 pictures for an exhibition at a gallery. 5 new galleries also … |
| 1246 | 929 | 271 | ✓ | A spaceship is traveling to another planet. The spaceship travels at a consistent … |
| 5986 | 889 | 246 | ✓ | Nancy wants to figure out if she can afford to apply to the University of Michigan … |

As três têm estrutura multi-etapa ("5 new galleries also" implica loop
implícito; problemas de taxa-tempo; orçamento com múltiplas despesas).
Em 2/3 o teacher acerta com CoT mais longo — não é patologia de
geração, é uso adequado do budget em problemas mais complexos.

---

## 7. Decisão

### **GO para Fase 1.1 com `max_length = 1024`**

### Justificativa

Os critérios substantivos do piloto passam:

1. **Formato perfeito.** `separator_rate = 1.000` em 100 amostras.
   Pipeline de masking region-aware em `src/data_gsm8k.py`
   (`text.find("####")`) funciona corretamente em 100% do output do
   teacher.
2. **Teacher competente.** Acurácia 0.92 supera por 12 pp o threshold
   A20, torna D3 trivialmente off.
3. **Budget dimensionado corretamente via D1.** O único gate fora
   (`p95 ≤ 768`) é precisamente o mecanismo que o v2.2 §4.5 e §7
   (A12) usam para decidir entre `max_length ∈ {768, 1024}`. A regra
   da Pré-0 — "upgrade para 1024 se truncamento > 15% em n grande"
   — é atingida (16.0% > 15% em 768), e 1024 absorve 100% da
   distribuição observada com folga de 68 tokens no pior caso.
4. **Sem truncamento mascarado.** `max_generation = 306` em
   `max_new_tokens = 512`; teacher nunca chega perto do teto, então a
   cauda de comprimento é sinal genuíno de complexidade dos problemas,
   não artefato de corte.

### Decisões de planejamento resolvidas

| # | Decisão | Resolução |
|---|---|---|
| D1 | `max_length` final (768 vs 1024) | **1024** — ver §3 e §7 acima |
| D3 | Filtrar `is_teacher_correct == False` do JSONL | **Não ativar** (`filter_teacher_wrong = false`) — ver §5 |

### Decisões pendentes (para fases posteriores, não bloqueantes)

| # | Decisão | Quando |
|---|---|---|
| D2 | Reduzir batch 16→8 (grad_accum 4→8) | Início Fase 4 se OOM em FKL bf16 |
| D6 | Ativar `enable_gradient_checkpointing` | Início Fase 4 se OOM persiste após D2 |

`max_length = 1024` é +33% em VRAM vs 768; D2/D6 permanecem como
mitigações previstas. Sem mudança além disso.

---

## 8. Próximo passo operacional

**Fase 1.1 — geração completa do CoT** (v2.2 §V.1.1):

- Implementar `scripts/generate_full_cot.py` como extensão do
  piloto, com mesmo system prompt / few-shot / decoding já validados.
- Gerar `data/gsm8k_cot_qwen25_7b.jsonl` com 7 473 train + 1 319 test
  = 8 792 linhas, `max_new_tokens = 512`, escrita linha a linha.
- Campos por linha conforme v2.2 §V.1.1 (split, idx, question,
  answer_gold, teacher_full_text, extracted_answer,
  is_teacher_correct, separator_found, total_len_tokens).
- Executar na A100 (~2–4 h GPU).

Implementação do script na próxima sessão do Claude Code.

---

**FIM — Fase 1.0 concluída. GO.**
