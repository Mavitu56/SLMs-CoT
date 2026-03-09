# Contexto do Projeto — KD Qwen2.5

## Descrição
Experimento de Knowledge Distillation (KD) em LLMs autoregressivos.
Compara Forward KL (FKL) vs Reverse KL (RKL) vs CE baseline em termos de
entropia token-level H(t), calibração (ECE) e divergência KL Teacher–Student.

- **Teacher:** Qwen/Qwen2.5-7B-Instruct
- **Student:** Qwen/Qwen2.5-1.5B-Instruct
- **Dataset treino:** databricks/databricks-dolly-15k
- **Dataset avaliação secundária:** cais/mmlu (split test)
- **Branch atual:** KD-Hinton → migrar para KD-Dolly

---

## Estado Atual do Repositório

### O que está correto e NÃO deve ser alterado
- `src/losses_kd.py` — `shift_for_causal_lm()`, `compute_ce()`, `compute_kd_forward_kl()` estão corretos
- `src/losses_kd.py` — `compute_total_loss()` já aplica `T²` explicitamente via `lambda_kd * (T**2) * loss_kd` ✔
- `src/train_manual.py` — `load_teacher()`, `load_student()`, verificação de vocabulário token-a-token
- `src/sanity.py` — todos os 6 checks existentes devem ser mantidos
- `src/utils_seed.py` — manter idêntico
- `src/evaluate_probabilistic.py` — estrutura de avaliação por região
- `src/analysis_metrics.py` — `compute_ece()` (verificar se é equal-width 10 bins antes de usar)

### O que precisa ser criado ou modificado
| Arquivo | Ação | Motivo |
|---|---|---|
| `src/losses_kd.py` | MODIFICAR: adicionar `compute_kd_reverse_kl()` e atualizar `compute_total_loss()` com `kd_mode` | RKL não existe ainda |
| `src/data_dolly.py` | CRIAR | Dataset atual é GSM8K; projeto usa Dolly-15k |
| `src/data_mmlu.py` | CRIAR | Avaliação secundária em MMLU |
| `src/sanity.py` | EXPANDIR: adicionar checks 7–11 | Cobrir RKL e ECE analítico |
| `src/train_manual.py` | ADAPTAR: ler `kd_mode` do config; adicionar cosine scheduler | Suporte às 3 condições |
| `configs/` | CRIAR 27 YAMLs (3 condições × 3 T × 3 seeds) | Cada run tem sua config |
| `tests/test_losses_numeric.py` | CRIAR | Validação numérica em distribuições toy |

---

## Fase Atual: Fase 3 — Formalização Matemática Interna

**Objetivo:** Garantir que todas as fórmulas estão implementadas corretamente ANTES de qualquer treino real.

### Ordem de Implementação (não pular etapas)
1. Adicionar `compute_kd_reverse_kl()` em `src/losses_kd.py`
2. Atualizar `compute_total_loss()` com parâmetro `kd_mode`
3. Criar `tests/test_losses_numeric.py` com casos analíticos
4. Criar `src/data_dolly.py` com masking `[/INST]`
5. Verificar `compute_ece()` contra caso analítico
6. Adicionar checks 7–11 em `src/sanity.py`
7. Adaptar `src/train_manual.py` com cosine scheduler e `kd_mode` dispatch
8. Criar 27 configs YAML
9. Criar `src/data_mmlu.py`
10. Rodar sanity completo e confirmar `ALL SANITY CHECKS PASSED`

---

## Especificação das Losses

### Notação
- `z_T`, `z_S`: logits do teacher e student, shape `[B, L, V]`
- `y`: labels, shape `[B, L]`, com `-100` nas posições ignoradas
- `T`: temperatura (escalar, T > 0)
- `M`: conjunto de posições válidas após causal shift e masking (`labels != -100` AND `attention_mask == 1`)
- `|M|`: número de tokens válidos (normalizador)

### Condição 1 — CE Baseline
```
L_CE = -(1/|M|) * Σ_{(b,t)∈M}  log p_S(y_{b,t} | y_{<t}, x)
```
- Sem temperatura. Sem T². Logits crus do student.
- `compute_ce()` já existe e está correto — não alterar.
- Para `kd_mode='ce_only'`: `L = L_CE`

### Condição 2 — Forward KL (Hinton 2015 + OKD Eq.2)
```
L_FKL = (1/|M|) * Σ_{(b,t)∈M}  KL(p_T^(T)(·|ctx) || p_S^(T)(·|ctx))

KL(p_T || p_S) = Σ_v  p_T(v) * [log p_T(v) - log p_S(v)]
```
- Softmax aplicado com temperatura T em AMBOS teacher e student.
- `compute_kd_forward_kl()` já existe e está correto — não alterar.
- Loss combinada: `L = alpha * L_CE + (1 - alpha) * T² * L_FKL`
- **O fator T² é aplicado na combinação, NÃO dentro de `compute_kd_forward_kl()`.**

### Condição 3 — Reverse KL (MiniLLM, Gu et al. 2024) ← CRIAR
```
L_RKL = (1/|M|) * Σ_{(b,t)∈M}  KL(p_S(·|ctx) || p_T(·|ctx))

KL(p_S || p_T) = Σ_v  p_S(v) * [log p_S(v) - log p_T(v)]
```
- **A ordem é INVERTIDA em relação ao FKL: p_S está à esquerda.**
- **SEM temperatura:** usar logits crus (T=1 implícito). Não dividir por T.
- **SEM fator T²** na loss combinada.
- Loss combinada: `L = alpha * L_CE + (1 - alpha) * L_RKL`

### Implementação de `compute_kd_reverse_kl()` — adicionar em `src/losses_kd.py`
```python
def compute_kd_reverse_kl(
    shift_student_logits: torch.Tensor,   # [B, L-1, V]
    shift_teacher_logits: torch.Tensor,   # [B, L-1, V]
    valid_mask: torch.Tensor,             # [B, L-1] bool
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reverse KL(p_S || p_T) averaged over valid tokens.
    Source: MiniLLM (Gu et al. 2024).
    NO temperature. NO T² factor. Normalised by number of valid tokens.
    """
    assert shift_student_logits.shape == shift_teacher_logits.shape
    B, L_minus1, V = shift_student_logits.shape
    assert valid_mask.shape == (B, L_minus1)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        return shift_student_logits.new_tensor(0.0)

    # T=1: no temperature division
    s_logits = shift_student_logits.float()
    t_logits = shift_teacher_logits.float()

    p_student   = F.softmax(s_logits, dim=-1)           # [B, L-1, V]
    logp_student = F.log_softmax(s_logits, dim=-1)      # [B, L-1, V]
    logp_teacher = torch.log(
        F.softmax(t_logits, dim=-1).clamp_min(eps)      # [B, L-1, V]
    )

    # KL(p_S || p_T) = Σ_v p_S(v) * [log p_S(v) - log p_T(v)]
    kl_per_token = (p_student * (logp_student - logp_teacher)).sum(dim=-1)  # [B, L-1]

    kl_masked = (kl_per_token * valid_mask.float()).sum()
    return kl_masked / n_valid.float()
```

### `compute_total_loss()` atualizada — substituir em `src/losses_kd.py`
```python
def compute_total_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    T: float,
    alpha: float,     # fixo em 0.5 conforme Fase 0
    kd_mode: str,     # 'ce_only' | 'fkl' | 'rkl'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns: (loss_total, loss_ce, loss_kd, n_valid_tokens)

    Modes:
      ce_only : L = L_CE
      fkl     : L = alpha * L_CE + (1 - alpha) * T² * L_FKL
      rkl     : L = alpha * L_CE + (1 - alpha) * L_RKL  (no T²)
    """
    shift_t, shift_labels, valid_mask = shift_for_causal_lm(
        teacher_logits, labels, attention_mask
    )
    shift_s, _, _ = shift_for_causal_lm(
        student_logits, labels, attention_mask
    )

    loss_ce = compute_ce(shift_s, shift_labels, valid_mask)

    if kd_mode == 'ce_only':
        loss_kd = shift_s.new_tensor(0.0)
        loss_total = loss_ce

    elif kd_mode == 'fkl':
        loss_kd = compute_kd_forward_kl(shift_t, shift_s, valid_mask, T)
        loss_total = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kd

    elif kd_mode == 'rkl':
        loss_kd = compute_kd_reverse_kl(shift_s, shift_t, valid_mask)
        loss_total = alpha * loss_ce + (1 - alpha) * loss_kd  # NO T²

    else:
        raise ValueError(f'kd_mode must be ce_only | fkl | rkl, got {kd_mode!r}')

    n_valid = int(valid_mask.sum().item())
    return loss_total, loss_ce, loss_kd, n_valid
```

---

## Especificação das Métricas

Todas as métricas são calculadas com **T=1 (logits crus)**, em **eval mode**, **SOMENTE sobre tokens da região de resposta** (após `[/INST]`), **ANTES de temperature scaling**.

### Entropia Token-Level H(t)
```
H(t) = - Σ_{v ∈ V}  p_S(v | y_{<t}, x) · log p_S(v | y_{<t}, x)

Onde p_S = softmax(z_S)  ← T=1
```
- Verificação: distribuição uniforme → H = log(V) ≈ 11.93 nats para V=152064
- Verificação: distribuição one-hot → H = 0

### ECE (Expected Calibration Error)
```
ECE = Σ_{m=1}^{10}  (|B_m| / n) * |acc(B_m) - conf(B_m)|
```
- **10 bins equal-width** no intervalo [0, 1]: B_1=[0.0,0.1), ..., B_10=[0.9,1.0]
- **Top-1 apenas:** `conf = max_v p_S(v|ctx)`, `acc = 1 se argmax == y_{b,t} else 0`
- **NÃO usar equal-mass** (Kadavath usa equal-mass — diferente do projeto)
- Calculado **ANTES de temperature scaling**

### KL Teacher–Student (métrica de avaliação)
```
KL(T||S) = Σ_{v ∈ V}  p_T(v|ctx) · log[ p_T(v|ctx) / p_S(v|ctx) ]
```
- **T=1** (logits crus) — esta é a métrica de avaliação, não a loss de treino
- Média sobre tokens válidos da região de resposta

### Masking [/INST]
```python
# Tokenizar texto completo e prompt separadamente
prompt_len = len(tokenizer(prompt_text).input_ids)
labels[:prompt_len] = -100          # prompt ignorado na loss
response_mask = (position >= prompt_len)  # apenas resposta para métricas

# Exemplos sem [/INST] → descartar e registrar como taxa de falha
```

---

## Alertas Críticos de Implementação

| ID | Regra | Consequência se violada |
|---|---|---|
| **A7** | Inserir `T²` manualmente na combinação FKL: `(T**2) * loss_kd`. Bibliotecas NÃO incluem T² automaticamente. | Loss FKL incorreta, resultados inválidos |
| **A5** | Normalizar por token `÷|M|` em todas as losses. OKD normaliza por sequência (diferente). | Valores de KL não comparáveis com OKD |
| **A9** | OKD NÃO tem ECE formal. Usa STD dos logits como proxy. Não usar como referência de calibração. | Comparação incorreta |
| **A1** | ECE com **equal-width** bins, não equal-mass (Kadavath usa equal-mass). | ECE não replicável |
| **A2** | ECE do MiniLLM (RKL=0.099 vs FKL=0.191) foi medido em SST2 (classificação). **Proibido usar como baseline numérico para Dolly.** | Comparação inválida |

---

## Sanity Checks a Adicionar (checks 7–11)

### Check 7 — RKL ≈ 0 quando teacher == student
```python
# Carregar dois modelos idênticos, calcular RKL
# Esperar: loss_rkl < 1e-2
```

### Check 8 — RKL é invariante à temperatura
```python
# RKL com T=1 deve ser IGUAL a RKL com T=4
# (RKL ignora o parâmetro T — usa logits crus)
assert torch.allclose(loss_rkl_T1, loss_rkl_T4, atol=1e-5)
```

### Check 9 — FKL varia com temperatura
```python
# FKL com T=1 deve ser DIFERENTE de FKL com T=4
assert not torch.allclose(loss_fkl_T1, loss_fkl_T4, atol=1e-3)
```

### Check 10 — ECE analítico
```python
# Modelo perfeitamente calibrado: conf(B_m) == acc(B_m) para todo m → ECE = 0
# Modelo sempre confiante (conf=1, acc=0.5) → ECE = 0.5
```

### Check 11 — Masking [/INST] funcional
```python
# Criar exemplo sintético com [/INST] em posição conhecida
# Verificar response_mask == True apenas para tokens de resposta
# Verificar que exemplos sem [/INST] são descartados e contabilizados
```

---

## Testes Numéricos Analíticos — `tests/test_losses_numeric.py`

```python
# Teste 1: KL(p||p) = 0 para FKL e RKL
loss_fkl_self = compute_kd_forward_kl(t_logits, t_logits, mask, T=1.0)
assert loss_fkl_self.item() < 1e-5

loss_rkl_self = compute_kd_reverse_kl(t_logits, t_logits, mask)
assert loss_rkl_self.item() < 1e-5

# Teste 2: KL(p||q) > 0 para p != q
assert compute_kd_forward_kl(t_logits, s_logits, mask, T=1.0).item() > 0
assert compute_kd_reverse_kl(s_logits, t_logits, mask).item() > 0

# Teste 3: Normalização por token (loss independe do batch size)
# Dobrar o batch com exemplos idênticos não deve alterar a loss
assert torch.allclose(loss_single, loss_double, atol=1e-5)

# Teste 4: RKL invariante a T
loss_rkl_T1 = compute_total_loss(..., T=1, kd_mode='rkl')[0]
loss_rkl_T4 = compute_total_loss(..., T=4, kd_mode='rkl')[0]
assert torch.allclose(loss_rkl_T1, loss_rkl_T4, atol=1e-5)

# Teste 5: FKL sensível a T
loss_fkl_T1 = compute_total_loss(..., T=1, kd_mode='fkl')[0]
loss_fkl_T4 = compute_total_loss(..., T=4, kd_mode='fkl')[0]
assert not torch.allclose(loss_fkl_T1, loss_fkl_T4, atol=1e-3)
```

---

## Parâmetros de Treino Congelados

| Parâmetro | Valor | Fonte |
|---|---|---|
| optimizer | AdamW | DA-KD, EasyDistill |
| learning_rate | 2e-5 | EasyDistill (Qwen2.5) |
| scheduler | cosine decay | DA-KD, EasyDistill |
| warmup_ratio | 0.1 | EasyDistill |
| weight_decay | 0.05 | EasyDistill |
| num_epochs | 3 | EasyDistill |
| effective_batch_size | 8 | DA-KD |
| max_length | 512 tokens | Fase 0 + DA-KD |
| alpha (α) | 0.5 fixo | Hinton 2015, Stanton 2021 |
| T sweep | {1, 2, 4} | OKD, Stanton, KD(C) |
| seeds | 42, 123, 7 | Fase 0 |

**Total de runs: 3 condições × 3 temperaturas × 3 seeds = 27 runs**

---

## Estrutura de Config YAML

```yaml
# Exemplo: configs/kd_dolly_fkl_T2_seed42.yaml
teacher_name: Qwen/Qwen2.5-7B-Instruct
student_name: Qwen/Qwen2.5-1.5B-Instruct
teacher_load_mode: 4bit
student_dtype: bf16

dataset: dolly          # novo campo — data_dolly.py
kd_mode: fkl            # ce_only | fkl | rkl
temperature: 2          # T ∈ {1, 2, 4}
alpha: 0.5              # fixo
seed: 42                # 42 | 123 | 7

max_length: 512
num_epochs: 3
batch_size: 2
grad_accum_steps: 4     # batch efetivo = 8
lr: 2.0e-5
weight_decay: 0.05
warmup_ratio: 0.1
scheduler: cosine
max_grad_norm: 1.0

log_every: 50
save_dir: checkpoints/fkl_T2_seed42
log_file: logs/fkl_T2_seed42.jsonl
run_sanity: true
```

---

## Regras Gerais

- Não adicionar variáveis novas sem justificativa extraída da literatura
- Não expandir escopo além do descrito neste arquivo
- Sempre registrar seeds e estatísticas (média ± DP sobre 3 seeds)
- Documentar cada decisão com referência à fase em que foi tomada
- Ao terminar cada item da ordem de implementação, rodar o teste correspondente antes de avançar
