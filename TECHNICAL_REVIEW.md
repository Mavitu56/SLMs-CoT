# Technical Review — KD Qwen2.5 Repository

**Data:** 2026-03-11
**Branch:** KD-Hinton
**Escopo:** Auditoria função-a-função de `src/losses_kd.py`, `src/analysis_metrics.py`, `src/data_dolly.py`, `src/train_manual.py` e `src/sanity.py`.

---

## 1. `src/losses_kd.py`

### shift_for_causal_lm()

**Arquivo:** `src/losses_kd.py:46`

**O que faz:**
Aplica o deslocamento padrão de next-token prediction para modelos causais. O logit na posição `t` deve prever o token na posição `t+1`. Além disso, constrói uma máscara booleana indicando quais posições são válidas para o cálculo da loss (posições onde o label não é `-100` E a attention mask é `1`).

**Fórmula implementada:**

```
shift_logits  = logits[:, :-1, :]          (posições 0..L-2)
shift_labels  = labels[:, 1:]              (posições 1..L-1)
shift_mask    = attention_mask[:, 1:]       (posições 1..L-1)

valid_mask(b,t) = shift_mask(b,t) == 1  AND  shift_labels(b,t) ≠ -100
```

**Inputs:**
| Parâmetro | Shape | Tipo |
|---|---|---|
| `logits` | `[B, L, V]` | float |
| `labels` | `[B, L]` | long |
| `attention_mask` | `[B, L]` | int/bool |

**Outputs:**
| Retorno | Shape | Tipo |
|---|---|---|
| `shift_logits` | `[B, L-1, V]` | float |
| `shift_labels` | `[B, L-1]` | long |
| `valid_mask` | `[B, L-1]` | bool |

**Decisões de implementação:**
- O shift da attention_mask é feito com `[:, 1:]` (alinhado com os labels, posição 1..L-1), e não com `[:, :-1]`. Isso é correto: a máscara acompanha as posições dos labels alvo.
- `.contiguous()` é chamado em todos os tensores de saída para garantir layout de memória contíguo.
- Asserts verificam dimensões e consistência de tamanhos entre logits, labels e attention_mask.

**Consistência com CLAUDE.md:** ✔ consistente. A especificação define M como o conjunto de posições válidas após causal shift com `labels != -100` AND `attention_mask == 1`. A implementação replica isso fielmente.

---

### compute_ce()

**Arquivo:** `src/losses_kd.py:78`

**O que faz:**
Calcula a cross-entropy padrão (hard-label) do student contra os labels verdadeiros, normalizada pelo número de tokens válidos. Não usa temperatura.

**Fórmula implementada:**

```
L_CE = -(1/|M|) * Σ_{(b,t)∈M}  log p_S(y_{b,t} | y_{<t}, x)

Onde  p_S = softmax(z_S)  com T=1 (logits crus)
|M| = número de tokens válidos (valid_mask == True)
```

Expandindo: usa `F.cross_entropy` internamente, que calcula `-log_softmax(logits)[label]` por posição, depois aplica masking e normaliza por `n_valid`.

**Inputs:**
| Parâmetro | Shape |
|---|---|
| `shift_student_logits` | `[B, L-1, V]` |
| `shift_labels` | `[B, L-1]` |
| `valid_mask` | `[B, L-1]` bool |

**Outputs:**
| Retorno | Shape |
|---|---|
| `loss_ce` | escalar |

**Decisões de implementação:**
- Flatten em `[B*(L-1), V]` e `[B*(L-1)]` antes de chamar `F.cross_entropy(reduction="none")`.
- `F.cross_entropy` do PyTorch já faz o cálculo em float32 internamente quando input é float32; para bf16/fp16, ocorre promoção automática no log_softmax.
- Se `n_valid == 0`, retorna tensor com valor `0.0` (sem divisão por zero, sem NaN).
- Normalização é **por token** (`÷ n_valid`), não por sequência.

**Consistência com CLAUDE.md:** ✔ consistente. A fórmula `-(1/|M|) * Σ log p_S(y)` está implementada corretamente. Sem temperatura. O CLAUDE.md diz "compute_ce() já existe e está correto — não alterar", e a implementação é fiel.

---

### compute_kd_forward_kl()

**Arquivo:** `src/losses_kd.py:111`

**O que faz:**
Calcula a Forward KL divergence KL(p_T || p_S) em distribuições suavizadas por temperatura T, normalizada pelo número de tokens válidos.

**Fórmula implementada:**

```
p_T = softmax(z_T / T)
p_S = softmax(z_S / T)

KL(p_T || p_S) = Σ_v  p_T(v) * [log p_T(v) - log p_S(v)]

L_FKL = (1/|M|) * Σ_{(b,t)∈M}  KL(p_T(·|ctx) || p_S(·|ctx))
```

**NOTA:** O fator T² **NÃO** aparece dentro desta função. Ele é aplicado externamente em `compute_total_loss()`.

**Inputs:**
| Parâmetro | Shape |
|---|---|
| `shift_teacher_logits` | `[B, L-1, V]` |
| `shift_student_logits` | `[B, L-1, V]` |
| `valid_mask` | `[B, L-1]` bool |
| `T` | escalar float (temperatura) |
| `eps` | escalar float (default `1e-8`) |

**Outputs:**
| Retorno | Shape |
|---|---|
| `loss_kd` | escalar |

**Decisões de implementação:**
- **Cast para float32:** `shift_teacher_logits.float()` e `shift_student_logits.float()` — garante estabilidade numérica quando modelos operam em bf16.
- **Divisão por T:** ambos os logits são divididos por `T` antes do softmax: `t_logits / T`, `s_logits / T`.
- **Cálculo de `p_teacher`:** via `F.softmax(t_logits, dim=-1)`.
- **Cálculo de `logp_teacher`:** via `torch.log(p_teacher.clamp_min(eps))` — **não usa** `F.log_softmax`. Usa clamp com `eps=1e-8` para evitar `log(0)`.
- **Cálculo de `logp_student`:** via `F.log_softmax(s_logits, dim=-1)` — **usa** log_softmax diretamente (mais estável numericamente).
- **Assimetria no cálculo:** log do teacher usa `log(softmax().clamp_min(eps))` enquanto log do student usa `log_softmax()`. Isso é intencional: `log_softmax` é numericamente mais estável, e como o gradiente flui pelo student (não pelo teacher), é mais importante que o student tenha precisão numérica. Para o teacher (detached/no_grad), o clamp é suficiente.
- **Normalização por token:** `÷ n_valid`, não por sequência.
- **Retorno `0.0` se `n_valid == 0`.**

**Consistência com CLAUDE.md:** ✔ consistente. A fórmula `KL(p_T || p_S)` com softmax em temperatura T, normalizada por `|M|`, sem T² interno, corresponde exatamente à especificação. O CLAUDE.md nota "compute_kd_forward_kl() já existe e está correto — não alterar".

---

### compute_kd_reverse_kl()

**Arquivo:** `src/losses_kd.py:151`

**O que faz:**
Calcula a Reverse KL divergence KL(p_S || p_T) — a ordem é **invertida** em relação ao FKL. Usa logits crus (T=1 implícito). Não aplica temperatura. Não aplica fator T².

**Fórmula implementada:**

```
p_S = softmax(z_S)          (T=1, sem divisão por temperatura)
p_T = softmax(z_T)          (T=1, sem divisão por temperatura)

KL(p_S || p_T) = Σ_v  p_S(v) * [log p_S(v) - log p_T(v)]

L_RKL = (1/|M|) * Σ_{(b,t)∈M}  KL(p_S(·|ctx) || p_T(·|ctx))
```

**Inputs:**
| Parâmetro | Shape | Nota |
|---|---|---|
| `shift_student_logits` | `[B, L-1, V]` | **primeiro argumento** — student primeiro (diferente do FKL) |
| `shift_teacher_logits` | `[B, L-1, V]` | segundo argumento |
| `valid_mask` | `[B, L-1]` bool | |
| `eps` | escalar float (default `1e-8`) | |

**Outputs:**
| Retorno | Shape |
|---|---|
| `loss_rkl` | escalar |

**Decisões de implementação:**
- **Sem temperatura:** nenhuma divisão por T nos logits. Usa logits crus diretamente.
- **Ordem dos argumentos invertida:** `(student, teacher)` — reflecte que p_S é a distribuição "fonte" em KL(p_S || p_T). O docstring documenta isso explicitamente.
- **Cast para float32:** ambos os logits são convertidos com `.float()`.
- **`p_student`:** via `F.softmax(s_logits, dim=-1)`.
- **`logp_student`:** via `F.log_softmax(s_logits, dim=-1)` — estabilidade numérica.
- **`logp_teacher`:** via `torch.log(F.softmax(t_logits, dim=-1).clamp_min(eps))` — mesma estratégia do FKL para o lado detached.
- **Normalização por token:** `÷ n_valid`.
- **Retorno `0.0` se `n_valid == 0`.**

**Consistência com CLAUDE.md:** ✔ consistente. A especificação define RKL como `KL(p_S || p_T)` sem temperatura, sem T², normalizado por `|M|`. A assinatura `(student, teacher)` é exatamente a documentada no CLAUDE.md. A implementação segue o pseudocódigo fornecido token-a-token.

---

### compute_total_loss()

**Arquivo:** `src/losses_kd.py:195`

**O que faz:**
Orquestra o cálculo da loss total combinando CE com a componente KD de acordo com o modo (`ce_only`, `fkl`, `rkl`).

**Fórmulas implementadas por modo:**

```
ce_only:   L = L_CE                                       (sem KD)
fkl:       L = α · L_CE + (1 − α) · T² · L_FKL           (com T²)
rkl:       L = α · L_CE + (1 − α) · L_RKL                (sem T²)
```

**Inputs:**
| Parâmetro | Shape/Tipo |
|---|---|
| `teacher_logits` | `[B, L, V]` (pode ser vazio para `ce_only`) |
| `student_logits` | `[B, L, V]` |
| `labels` | `[B, L]` |
| `attention_mask` | `[B, L]` |
| `T` | escalar float |
| `alpha` | escalar float (fixo em 0.5) |
| `kd_mode` | string: `'ce_only'` \| `'fkl'` \| `'rkl'` |

**Outputs:**
| Retorno | Shape/Tipo |
|---|---|
| `loss_total` | escalar |
| `loss_ce` | escalar |
| `loss_kd` | escalar (`0.0` para `ce_only`) |
| `n_valid_tokens` | int |

**Decisões de implementação:**
- **Causal shift duplo:** chama `shift_for_causal_lm()` para o student primeiro (sempre), depois para o teacher apenas nos modos `fkl`/`rkl`. Isso implica que para `ce_only`, o teacher forward pode ser omitido (e de facto é omitido no `train_manual.py`).
- **T² aplicado explicitamente:** `(1 - alpha) * (T ** 2) * loss_kd` no modo `fkl`. Isso segue Hinton 2015 e o alerta A7 do CLAUDE.md.
- **T² ausente no RKL:** `(1 - alpha) * loss_kd` sem T². Conforme MiniLLM (Gu et al. 2024).
- **Ordem dos argumentos para RKL:** `compute_kd_reverse_kl(shift_s, shift_t, ...)` — student primeiro, teacher segundo. Correto.
- **ValueError** para `kd_mode` inválido.

**Consistência com CLAUDE.md:** ⚠ divergência menor na estrutura (sem impacto no resultado).

O CLAUDE.md especifica na seção `compute_total_loss() atualizada`:
```python
shift_t, shift_labels, valid_mask = shift_for_causal_lm(
    teacher_logits, labels, attention_mask
)
shift_s, _, _ = shift_for_causal_lm(
    student_logits, labels, attention_mask
)
```

A implementação real faz o **inverso** — aplica o shift no **student primeiro** e só faz o shift no teacher dentro de cada branch `elif`:
```python
shift_s, shift_labels, valid_mask = shift_for_causal_lm(
    student_logits, labels, attention_mask,
)
# ... dentro do elif:
shift_t, _, _ = shift_for_causal_lm(
    teacher_logits, labels, attention_mask,
)
```

**Impacto:** Nenhum. O `shift_for_causal_lm` é uma operação determinística de fatiamento; a ordem não altera os resultados. Além disso, esta versão é mais eficiente pois evita o shift do teacher no modo `ce_only`. Os resultados numéricos são idênticos.

---

## 2. `src/analysis_metrics.py`

### token_entropy() / mean_entropy()

**Arquivo:** `src/analysis_metrics.py:73` / `src/analysis_metrics.py:102`

**O que faz:**
Calcula a entropia token-level H(t) da distribuição preditiva do modelo.

**Fórmula implementada:**

```
H(t) = - Σ_{v ∈ V}  p(v) · log p(v)

Onde  p = softmax(z)  com T=1 (logits crus)

mean_entropy = (1/|M|) * Σ_{(b,t)∈M}  H(b,t)
```

**Inputs (token_entropy):**
| Parâmetro | Shape |
|---|---|
| `logits` | `[B, L, V]` (já com shift aplicado) |

**Outputs (token_entropy):**
| Retorno | Shape |
|---|---|
| `entropy` | `[B, L]` |

**Inputs (mean_entropy):**
| Parâmetro | Shape |
|---|---|
| `logits` | `[B, L, V]` |
| `valid_mask` | `[B, L]` bool |

**Outputs (mean_entropy):**
| Retorno | Shape |
|---|---|
| `mean_ent` | escalar |

**Decisões de implementação:**
- **Cast para float32:** `logits.float()`.
- **Reutilização de log_softmax:** calcula `log_probs = F.log_softmax(...)`, depois `probs = log_probs.exp()`. Isso reutiliza a identidade `softmax = exp(log_softmax)`, evitando dois passes pelo softmax.
- **Clamp final:** `entropy.clamp_min(0.0)` — entropia nunca é negativa; o clamp protege contra ruído de ponto flutuante.
- **Assert:** verifica que `entropy >= -1e-6` antes do clamp.
- **Sem temperatura:** logits crus (T=1).

**Consistência com CLAUDE.md:** ✔ consistente. A especificação define `H(t) = - Σ p_S(v) · log p_S(v)` com `p_S = softmax(z_S)` em T=1. Condições de contorno documentadas: uniforme → `log(V)`, one-hot → `0`.

---

### token_max_probability() / mean_max_probability()

**Arquivo:** `src/analysis_metrics.py:128` / `src/analysis_metrics.py:145`

**O que faz:**
Calcula a probabilidade máxima (sharpness) por token.

**Fórmula implementada:**

```
maxprob(t) = max_v  p(v|ctx)

Onde  p = softmax(z) com T=1

mean_maxprob = (1/|M|) * Σ_{(b,t)∈M}  maxprob(b,t)
```

**Inputs/Outputs:** Mesma estrutura de `token_entropy`/`mean_entropy`.

**Decisões de implementação:**
- Cast para float32.
- Assert em `mean_max_probability`: resultado deve estar em `[0, 1]`.

**Consistência com CLAUDE.md:** ✔ consistente. `maxprob_t = max_v p_t(v)` corresponde ao definido na especificação.

---

### kl_per_position() / mean_kl()

**Arquivo:** `src/analysis_metrics.py:174` / `src/analysis_metrics.py:231`

**O que faz:**
Calcula a Forward KL divergence KL(P^T || P^S) como **métrica de avaliação** (não loss de treino). Usa logits crus (T=1). `kl_per_position` retorna uma curva por posição; `mean_kl` retorna o escalar médio.

**Fórmula implementada:**

```
p_T = softmax(z_T)    (T=1, logits crus)
p_S = softmax(z_S)    (T=1, logits crus)

KL(P^T || P^S) = Σ_v  p_T(v) · [log p_T(v) - log p_S(v)]

mean_kl = (1/|M|) * Σ_{(b,t)∈M}  KL(p_T(b,t) || p_S(b,t))
```

**Inputs (mean_kl):**
| Parâmetro | Shape |
|---|---|
| `teacher_logits` | `[B, L, V]` (shifted) |
| `student_logits` | `[B, L, V]` (shifted) |
| `valid_mask` | `[B, L]` bool |
| `eps` | float (default `1e-8`) |

**Outputs:**
| Retorno | Shape |
|---|---|
| `kl_curve` | `[L]` (para `kl_per_position`) |
| `mean_kl` | escalar (para `mean_kl`) |

**Decisões de implementação:**
- **Sem temperatura:** logits crus. É uma métrica de avaliação, não a loss de treino.
- **Cast para float32:** ambos os logits.
- **`logp_teacher`:** `torch.log(p_teacher.clamp_min(eps))` com `eps=1e-8`.
- **`logp_student`:** `F.log_softmax(s_f32, dim=-1)` (mais estável).
- **Clamp de KL:** `kl_tokens.clamp_min(0.0)` após assert `>= -1e-6`.
- **`kl_per_position`:** média ao longo da dimensão batch para cada posição. Posições sem tokens válidos recebem 0 (via `safe_n = n_valid_per_pos.clamp_min(1.0)`).

**Consistência com CLAUDE.md:** ✔ consistente. A especificação define `KL(T||S) = Σ p_T(v) · log[p_T(v) / p_S(v)]` com T=1. A implementação é equivalente a `Σ p_T · (log_p_T - log_p_S)`, que é algebricamente idêntica.

---

### compute_ece()

**Arquivo:** `src/analysis_metrics.py:272`

**O que faz:**
Calcula o Expected Calibration Error (ECE) top-1 com bins equal-width.

**Fórmula implementada:**

```
ECE = Σ_{m=1}^{n_bins}  (|B_m| / N) * |acc(B_m) - conf(B_m)|

Onde:
  conf(b,t) = max_v  p(v|ctx)           (top-1 confidence)
  acc(b,t)  = 1 se argmax_v p(v) == y_{b,t},  0 caso contrário
  B_m       = bins equal-width em [0, 1]
  N         = número total de tokens válidos
```

**Bins:**
- `n_bins = 10` (default)
- Equal-width: `B_1=[0.0, 0.1)`, `B_2=[0.1, 0.2)`, ..., `B_10=[0.9, 1.0]`
- Último bin é **inclusive** em ambos os lados: `[0.9, 1.0]`

**Inputs:**
| Parâmetro | Shape |
|---|---|
| `logits` | `[B, L, V]` (shifted) |
| `labels` | `[B, L]` (shifted, ground-truth) |
| `valid_mask` | `[B, L]` bool |
| `n_bins` | int (default 10) |

**Outputs:**
| Retorno | Shape |
|---|---|
| `ece` | escalar em [0, 1] |

**Decisões de implementação:**
- **Cast para float32:** `logits.float()` no softmax.
- **Top-1 apenas:** `max_probs, preds = probs.max(dim=-1)`.
- **Equal-width bins:** via `torch.linspace(0.0, 1.0, n_bins + 1)`.
- **Último bin inclusivo:** `(flat_conf >= lo) & (flat_conf <= hi)` para `i == n_bins - 1`, enquanto os demais bins usam `(flat_conf >= lo) & (flat_conf < hi)`. Isso evita perder tokens com confiança exatamente 1.0.
- **Cálculo ANTES de temperature scaling:** os logits devem ser crus (T=1). A função não aplica nenhuma temperatura.
- Assert que o resultado está em `[0, 1]` (com tolerância `1e-6`).
- Retorna `0.0` se não há tokens válidos.

**Consistência com CLAUDE.md:** ✔ consistente. O CLAUDE.md especifica:
- 10 bins equal-width (✔)
- Top-1 apenas (✔)
- Antes de temperature scaling (✔)
- Não usar equal-mass (Kadavath) (✔ — usa equal-width)
- Alerta A1 cumprido.

---

### mean_nll()

**Arquivo:** `src/analysis_metrics.py:422`

**O que faz:**
Calcula a Negative Log-Likelihood média sobre tokens válidos.

**Fórmula implementada:**

```
NLL(b,t) = -log p(y_{b,t} | ctx)

Onde p = softmax(z) com T=1

mean_NLL = (1/|M|) * Σ_{(b,t)∈M}  NLL(b,t)
```

**Inputs:**
| Parâmetro | Shape |
|---|---|
| `logits` | `[B, L, V]` (shifted) |
| `labels` | `[B, L]` (shifted) |
| `valid_mask` | `[B, L]` bool |

**Outputs:**
| Retorno | Shape |
|---|---|
| `mean_nll` | escalar |

**Decisões de implementação:**
- **Cast para float32:** `logits.float()` no `token_nll`.
- **Safe gather:** labels com valor `-100` são substituídos por `0` via `clamp_min(0)` antes do `gather`. Isso evita erros de indexação; as posições com label `-100` são excluídas pela `valid_mask` posteriormente.
- **Retorna `0.0` se `n_valid == 0`.**
- **Perplexity:** `perplexity = exp(mean_nll)` — implementada separadamente em `perplexity()` (linha 447).

**Consistência com CLAUDE.md:** ✔ consistente. A fórmula `NLL_t = -log p_t(y_t)` e `Mean NLL = (1/|M|) Σ NLL_t` correspondem à especificação.

---

## 3. `src/data_dolly.py`

### load_dolly()

**Arquivo:** `src/data_dolly.py:25`

**O que faz:**
Carrega o dataset `databricks/databricks-dolly-15k` do Hugging Face.

**Decisões de implementação:**
- O dataset possui apenas o split `train`. Para avaliação, deve-se usar `train_test_split()` externamente.

**Consistência com CLAUDE.md:** ✔ consistente. O dataset especificado no CLAUDE.md é `databricks/databricks-dolly-15k`.

---

### tokenise_example()

**Arquivo:** `src/data_dolly.py:39`

**O que faz:**
Tokeniza um exemplo do Dolly usando o chat template do modelo. Constrói o prompt a partir de `instruction + context`, aplica masking no prompt (labels = -100), e retorna `input_ids`, `attention_mask` e `labels`.

**Procedimento detalhado:**

1. **Construção do prompt:**
   - Se `context` existe e não é vazio: `prompt_content = "{instruction}\n\n{context}"`
   - Se `context` é vazio: `prompt_content = instruction`

2. **Texto completo (chat template):**
   - Mensagens: `[{"role": "user", "content": prompt_content}, {"role": "assistant", "content": response}]`
   - Aplicação: `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`

3. **Determinação de `prompt_len`:**
   - Constrói mensagens somente com o prompt: `[{"role": "user", "content": prompt_content}]`
   - Aplica chat template **com** `add_generation_prompt=True` (inclui tokens de início de geração)
   - Tokeniza separadamente: `prompt_len = len(tokenizer(prompt_text).input_ids)`

4. **Masking:**
   - `labels[:prompt_len] = [-100] * prompt_len`
   - `prompt_len = min(prompt_len, len(labels))` — proteção contra truncation

5. **Exemplos sem resposta:**
   - Se `response` é vazio ou só whitespace: retorna `None`
   - O caller (`build_dataloader`) filtra esses casos e registra a taxa de descarte

**Inputs:**
| Parâmetro | Tipo |
|---|---|
| `example` | dict com keys `instruction`, `context`, `response` |
| `tokenizer` | `PreTrainedTokenizerBase` |
| `max_length` | int |

**Outputs:**
| Retorno | Tipo |
|---|---|
| dict ou None | `{"input_ids": list, "attention_mask": list, "labels": list}` |

**Decisões de implementação:**
- **Chat template nativo:** usa `apply_chat_template()` do tokenizer, não constrói tags manualmente (ex: `[/INST]`).
- **`add_special_tokens=False`** na tokenização — os special tokens já foram inseridos pelo chat template.
- **`attention_mask = [1] * len(input_ids)`** — toda posição recebe mask 1; padding é tratado no collator.
- **Truncação:** `truncation=True, max_length=max_length` no tokenizer.

**Consistência com CLAUDE.md:** ⚠ divergência na terminologia de masking.

O CLAUDE.md refere-se consistentemente a "masking `[/INST]`" como conceito de separação prompt/resposta. A implementação **não usa** o token literal `[/INST]` — usa o chat template nativo do Qwen2.5, que utiliza tokens diferentes (ex: `<|im_start|>assistant`). O mecanismo é equivalente: separa prompt de resposta via tokenização separada. A terminologia `[/INST]` no CLAUDE.md é herdada do formato Llama e é usada como conceito genérico, não como implementação literal. **O resultado funcional é idêntico ao especificado.**

---

### build_dataloader()

**Arquivo:** `src/data_dolly.py:123`

**O que faz:**
Pipeline completo: carrega Dolly → tokeniza → filtra exemplos vazios → calcula estatísticas de truncação → cria DataLoader.

**Decisões de implementação:**
- Usa `KDCollator` importado de `src/data_gsm8k.py` para padding.
- `num_workers=4`, `pin_memory=True`, `persistent_workers=True`.
- `drop_last=False`.
- `micro_overfit_n`: se definido, limita o dataset aos primeiros N exemplos (para debug/overfit).
- Loga: total de exemplos, comprimento médio, taxa de truncação, taxa de descarte de exemplos vazios.

**Consistência com CLAUDE.md:** ✔ consistente.

---

## 4. `src/train_manual.py`

### Estrutura do Loop de Treino

**Arquivo:** `src/train_manual.py:140`

**O que faz:**
Loop de treino epoch-based com gradient accumulation, cosine scheduler com warmup, e despacho de `kd_mode` para `compute_total_loss`.

**Estrutura:**

```
Para cada epoch (1..num_epochs):
    Para cada batch do DataLoader:
        1. Teacher forward (com torch.no_grad(), skip total para ce_only)
        2. Student forward
        3. Truncar vocab do teacher se necessário
        4. compute_total_loss(teacher_logits, student_logits, labels, mask, T, alpha, kd_mode)
        5. loss_total / grad_accum_steps → backward
        6. A cada grad_accum_steps micro-steps:
           a. clip_grad_norm_(max_grad_norm)
           b. optimizer.step() + scheduler.step()
           c. Logging a cada log_every optimizer steps
           d. Checkpoint a cada save_every optimizer steps
    Se há micro-steps residuais no fim do epoch:
        → optimizer.step() + scheduler.step() (flush de gradientes parciais)
```

**Decisões de implementação:**
- **Epoch-based, não step-based:** o critério de parada é `num_epochs`, não `num_steps`.
- **Gradient accumulation:** `scaled_loss = loss_total / grad_accum_steps`. Optimizer step a cada `grad_accum_steps` micro-batches.
- **Trailing micro-steps:** se `micro_count % grad_accum_steps != 0` ao fim de um epoch, faz um optimizer step com os gradientes acumulados. Isso garante que nenhum gradiente é desperdiçado.
- **Effective batch size:** `batch_size * grad_accum_steps` (ex: 2 * 4 = 8).
- **Skip teacher no ce_only:** não executa forward no teacher, cria tensor vazio `torch.empty(0, device=device)`.
- **Vocab truncation:** trunca logits do teacher para o tamanho do student se necessário. Assert de consistência.

### Cosine Scheduler com Warmup

**Arquivo:** `src/train_manual.py:220`

```
total_batches         = num_epochs * len(loader)
total_optimizer_steps = total_batches // grad_accum_steps
warmup_steps          = int(warmup_ratio * total_optimizer_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps,
)
```

**Decisões de implementação:**
- Usa `get_cosine_schedule_with_warmup` do HuggingFace Transformers.
- `warmup_ratio = 0.1` (default).
- O scheduler é stepped a cada optimizer step (não a cada micro-step).
- `scheduler.step()` é chamado imediatamente após `optimizer.step()`.

**Consistência com CLAUDE.md:** ✔ consistente. Os parâmetros `lr=2e-5`, `weight_decay=0.05`, `warmup_ratio=0.1`, `scheduler=cosine` correspondem à tabela de parâmetros congelados.

### Despacho de kd_mode

**Arquivo:** `src/train_manual.py:300`

A função `compute_total_loss` é chamada diretamente com `kd_mode=kd_mode` vindo da config. Todo o despacho (ce_only/fkl/rkl) é feito internamente por `compute_total_loss`.

**Consistência com CLAUDE.md:** ✔ consistente.

### Logging e o Fator T²

**Arquivo:** `src/train_manual.py:331`

Os valores logados são:
- `loss_total`: a loss combinada (que inclui T² para FKL, sem T² para RKL)
- `loss_ce`: a CE pura
- `loss_kd`: a componente KD pura (**sem** T²)

**Nota:** O `loss_kd` logado é o valor retornado por `compute_kd_forward_kl` ou `compute_kd_reverse_kl` diretamente — **antes** da multiplicação por T². O `loss_total` já inclui o T² no FKL. Isso é relevante ao interpretar os logs: para reconstruir `loss_total` a partir de `loss_ce` e `loss_kd` nos logs de FKL, é necessário aplicar `alpha * ce + (1-alpha) * T² * kd`.

**Consistência com CLAUDE.md:** ✔ consistente. O CLAUDE.md define que T² é aplicado na combinação, e os logs refletem isso fielmente.

### Validação de Config

**Arquivo:** `src/train_manual.py:144`

- Rejeita configs legadas com `lambda_kd`/`lambda_ce` (formato antigo).
- Rejeita configs com `num_steps` sem `num_epochs`.

---

## 5. `src/sanity.py`

### Visão Geral dos Checks

| Check | Função | O que verifica |
|---|---|---|
| 6.1 | `check_teacher_frozen()` | Teacher não acumula gradientes após backward |
| 6.2 | `check_kd_mode_combinations()` | Fórmula de combinação para cada modo (ce/fkl/rkl) |
| 6.3 | `check_kd_near_zero_same_model()` | FKL ≈ 0 quando teacher == student |
| 6.4 | `check_kl_non_negative()` | KL ≥ 0 |
| 6.5 | `check_mask_all_ignored()` | Labels todos -100 → loss = 0, sem NaN |
| 6.6 | `check_padding_attention_mask()` | Padding usa attention_mask=0 e labels=-100 |
| 7 | `check_rkl_near_zero_same_model()` | RKL ≈ 0 quando teacher == student |
| 8 | `check_rkl_temperature_invariant()` | RKL(T=1) == RKL(T=4) |
| 9 | `check_fkl_temperature_sensitive()` | FKL(T=1) ≠ FKL(T=4) |
| 10 | `check_ece_analytical()` | ECE analítico: uniforme ≈ 0, overconfiante ≈ 0.5 |
| 11 | `check_masking_prompt_len()` | Labels prompt = -100, resposta = input_ids, vazio → None |

---

### Check 6.1 — check_teacher_frozen()

**Arquivo:** `src/sanity.py:75`

**O que verifica matematicamente:**
Após um forward+backward do student com loss FKL, nenhum parâmetro do teacher deve ter `.grad` definido (deve ser `None`).

**Procedimento:**
1. Teacher em `eval()` + `torch.no_grad()`
2. Student forward + `compute_total_loss(kd_mode='fkl')`
3. `loss_total.backward()`
4. Loop sobre `teacher.named_parameters()`: assert `p.grad is None`

**Consistência com CLAUDE.md:** ✔ consistente.

---

### Check 6.2 — check_kd_mode_combinations()

**Arquivo:** `src/sanity.py:126`

**O que verifica matematicamente:**
Para cada modo, a loss total é reconstruída a partir das componentes:
- `ce_only`: `loss_total == loss_ce` e `loss_kd == 0`
- `fkl`: `loss_total == α·loss_ce + (1-α)·T²·loss_kd`
- `rkl`: `loss_total == α·loss_ce + (1-α)·loss_kd` (sem T²)

Usa `torch.allclose(atol=1e-5)`.

**Consistência com CLAUDE.md:** ✔ consistente.

---

### Check 6.3 — check_kd_near_zero_same_model()

**Arquivo:** `src/sanity.py:190`

**O que verifica:**
Carrega duas cópias independentes do mesmo modelo (`Qwen/Qwen2.5-0.5B-Instruct`). Calcula FKL entre elas. Espera `loss_kd < 1e-2`.

**Thresholds:**
- Absoluto: `kd_val < 1e-2` (generoso para ruído bf16)
- Relativo (se `normal_kd_ref` fornecido): `kd_val / normal_kd_ref < 0.01` (self-KD deve ser ≥100x menor que KD real)

**Consistência com CLAUDE.md:** ✔ consistente. O CLAUDE.md especifica `loss_kd < 1e-2`.

---

### Check 6.4 — check_kl_non_negative()

**Arquivo:** `src/sanity.py:258`

**O que verifica:**
`loss_kd >= -1e-5` para FKL com teacher ≠ student.

**Consistência com CLAUDE.md:** ✔ consistente (propriedade matemática de KL).

---

### Check 6.5 — check_mask_all_ignored()

**Arquivo:** `src/sanity.py:297`

**O que verifica:**
Com todos os labels = -100: `n_valid == 0`, `loss_total == 0.0`, nenhum NaN.

**Consistência com CLAUDE.md:** ✔ consistente.

---

### Check 6.6 — check_padding_attention_mask()

**Arquivo:** `src/sanity.py:342`

**O que verifica:**
Com batch_size=4 (forçando padding), verifica que `labels == -100` nas posições onde `attention_mask == 0`.

**Consistência com CLAUDE.md:** ✔ consistente.

---

### Check 7 — check_rkl_near_zero_same_model()

**Arquivo:** `src/sanity.py:385`

**O que verifica matematicamente:**
Carrega duas cópias do mesmo modelo. Calcula RKL entre elas. Espera `loss_kd < 1e-2`.

**Procedimento:**
1. Carrega `model_a` e `model_b` — ambos `Qwen/Qwen2.5-0.5B-Instruct`
2. `compute_total_loss(..., kd_mode='rkl', T=1.0)`
3. Assert `kd_val < 1e-2`

**Consistência com CLAUDE.md:** ✔ consistente. CLAUDE.md check 7: "loss_rkl < 1e-2".

---

### Check 8 — check_rkl_temperature_invariant()

**Arquivo:** `src/sanity.py:432`

**O que verifica matematicamente:**
RKL não usa temperatura — portanto, a loss total no modo `rkl` deve ser **idêntica** para T=1 e T=4.

**Procedimento:**
1. Forward teacher e student (mesmos logits)
2. `loss_T1 = compute_total_loss(..., T=1.0, kd_mode='rkl')[0]`
3. `loss_T4 = compute_total_loss(..., T=4.0, kd_mode='rkl')[0]`
4. `torch.allclose(loss_T1, loss_T4, atol=1e-5)`

**Nota:** Compara `loss_total` (não `loss_kd`). Como `loss_total = α·loss_ce + (1-α)·loss_kd` e `loss_ce` não depende de T, e `loss_kd (RKL)` não depende de T, a loss total também é invariante.

**Consistência com CLAUDE.md:** ✔ consistente. CLAUDE.md check 8: "assert torch.allclose(loss_rkl_T1, loss_rkl_T4, atol=1e-5)".

---

### Check 9 — check_fkl_temperature_sensitive()

**Arquivo:** `src/sanity.py:475`

**O que verifica matematicamente:**
FKL usa temperatura no softmax e tem T² no fator de combinação — portanto, a loss total no modo `fkl` deve **diferir** para T=1 e T=4.

**Procedimento:**
1. Forward teacher e student (mesmos logits)
2. `loss_T1 = compute_total_loss(..., T=1.0, kd_mode='fkl')[0]`
3. `loss_T4 = compute_total_loss(..., T=4.0, kd_mode='fkl')[0]`
4. `assert not torch.allclose(loss_T1, loss_T4, atol=1e-3)`

**Consistência com CLAUDE.md:** ✔ consistente. CLAUDE.md check 9: "assert not torch.allclose(loss_fkl_T1, loss_fkl_T4, atol=1e-3)".

---

### Check 10 — check_ece_analytical()

**Arquivo:** `src/sanity.py:518`

**O que verifica matematicamente:**

**Caso 1 — distribuição uniforme:**
- Logits: zeros (`[1, 1000, 10]`) → `p(v) = 1/V = 0.1` para todo v
- Labels: aleatórios em `[0, V)` → `acc ≈ 1/V = 0.1`
- Confiança: `max_v p(v) = 0.1` → todos os tokens caem no bin `[0.0, 0.1]`
- ECE ≈ `|0.1 - 0.1| = 0.0` → assert `ece < 0.05`

**Caso 2 — modelo overconfiante:**
- Logits: classe 0 com logit 100, restante com 0 → `conf ≈ 1.0`
- Labels: metade classe 0 (corretas), metade classe 1 (erradas) → `acc = 0.5`
- Todos os tokens caem no bin `[0.9, 1.0]`
- ECE ≈ `|1.0 - 0.5| = 0.5` → assert `|ece - 0.5| < 0.05`

**Consistência com CLAUDE.md:** ✔ consistente. CLAUDE.md check 10: "Modelo perfeitamente calibrado → ECE = 0" e "Modelo sempre confiante (conf=1, acc=0.5) → ECE = 0.5".

---

### Check 11 — check_masking_prompt_len()

**Arquivo:** `src/sanity.py:555`

**O que verifica:**
1. Cria exemplo sintético: `instruction="What is the capital of France?"`, `response="The capital of France is Paris."`
2. Tokeniza via `tokenise_example()`
3. Verifica que todos os labels antes de `first_valid` são `-100` (prompt mascarado)
4. Verifica que existem labels válidos (tokens de resposta)
5. Verifica que labels válidos coincidem com `input_ids` (teacher forcing)
6. Verifica que exemplo com resposta vazia retorna `None`

**Consistência com CLAUDE.md:** ✔ consistente. CLAUDE.md check 11: verificar que `response_mask == True` apenas para tokens de resposta e que exemplos sem `[/INST]` (resposta) são descartados.

---

## 6. Resumo das Divergências

| Arquivo | Função | Tipo | Detalhe | Impacto |
|---|---|---|---|---|
| `src/losses_kd.py` | `compute_total_loss()` | Estrutural | Shift do student primeiro (não teacher); shift do teacher condicional por branch | **Nenhum.** Resultados numéricos idênticos. Mais eficiente para `ce_only`. |
| `src/data_dolly.py` | `tokenise_example()` | Terminológica | CLAUDE.md refere "masking `[/INST]`", implementação usa chat template Qwen2.5 (sem `[/INST]` literal) | **Nenhum.** O mecanismo é equivalente: tokenização separada do prompt para determinar boundary. |

Nenhuma divergência compromete a correção numérica ou a validade dos resultados experimentais.

---

## 7. Verificação Cruzada: Alertas Críticos do CLAUDE.md

| ID | Regra | Status |
|---|---|---|
| **A7** | T² aplicado manualmente na combinação FKL | ✔ `(T ** 2) * loss_kd` em `compute_total_loss` linha 234 |
| **A5** | Normalização por token `÷|M|` em todas as losses | ✔ `÷ n_valid` em `compute_ce`, `compute_kd_forward_kl`, `compute_kd_reverse_kl` |
| **A9** | OKD não tem ECE formal — não usar como referência | ✔ Não referenciado no código |
| **A1** | ECE com equal-width, não equal-mass | ✔ `torch.linspace(0, 1, n_bins+1)` em `compute_ece` |
| **A2** | ECE do MiniLLM em SST2 não é baseline para Dolly | ✔ Não referenciado no código |
