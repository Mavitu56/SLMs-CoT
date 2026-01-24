# Riscos Identificados e Ajustes Implementados

Este documento descreve os problemas identificados durante a auditoria do repositório e as correções aplicadas para execução estável em **Google Colab com GPU A100**.

---

## 1. Problemas Identificados no KD Logits

### 1.1 Explosão de Gradientes (Loss não-finita)
**Sintoma**: Mensagens frequentes de `"loss não-finita; pulando batch"` durante treino com Qwen 7B → 1.5B.

**Causas Raiz**:
| Causa | Impacto | Solução |
|-------|---------|---------|
| Temperature muito alta (5.0→3.0→1.0) | Amplifica KL divergence exponencialmente | Reduzido para [2.0, 1.5, 1.0] |
| Alpha muito alto (0.7→0.5→0.3) | Peso excessivo no KD loss instável | Reduzido para [0.5, 0.4, 0.3] |
| `max_logit_abs=100` | Permite logits extremos que causam softmax overflow | Reduzido para 50.0 |
| `clip_grad_norm=1.0` | Clipping insuficiente para gradientes explosivos | Reduzido para 0.5 |
| Batches com poucos tokens supervisionados | CE/KL com denominador pequeno → gradientes instáveis | Adicionado `min_supervised_tokens=8` |

### 1.2 Vocabulário Grande do Qwen (~150k tokens)
**Impacto**: Tensores de logits `(B, T, V)` consomem muita VRAM e podem causar OOM.

**Mitigação**:
- Batch size pequeno (4 para A100 80GB)
- Logits calculados on-the-fly (não cacheados em disco)
- Gradient checkpointing ativado por padrão

### 1.3 Mismatch de Vocabulário Teacher/Student
**Contexto**: Mesmo usando modelos da mesma família (Qwen), diferentes versões podem ter vocabulários levemente diferentes.

**Proteção Existente**:
- `_ensure_vocab_alignment()`: expande embeddings do student se necessário
- `_align_teacher_logits()`: trunca ou padda logits do teacher

---

## 2. Outros Riscos Identificados

### 2.1 Avaliação com Limites Muito Baixos
**Arquivo**: [config.py](config.py)
```python
eval_limit_gsm8k: int = 100
eval_limit_bbh: int = 30
```
**Risco**: Resultados com alta variância estatística, especialmente para análise por seed.

**Recomendação**: Para artigo científico, aumentar para:
- GSM8K: 500-1000 exemplos
- BBH: 100-200 exemplos por task

### 2.2 Prompt Masking + Truncation
**Arquivo**: [distill.py](distill.py#L60-L110)

**Risco**: Se `max_length` for muito curto, a completion pode ser truncada completamente, resultando em `labels=-100` para todos os tokens.

**Mitigação Implementada**: Guard que pula batches com `valid_tokens < min_supervised_tokens`.

### 2.3 CasCoD Two-Stage Training
**Arquivo**: [run_experiment.py](run_experiment.py#L370-L550)

**Risco**: O estágio 2 (q,r→a) assume que o rationale do estágio 1 está correto. Se o student gerar raciocínio errado, propaga o erro.

**Status**: Funciona como projetado; é uma limitação inerente ao método CasCoD.

### 2.4 Geração de CoT do Teacher
**Arquivo**: [teacher_generation.py](teacher_generation.py)

**Risco**: Se o teacher gerar CoT degenerado (loops, vazio, sem resposta), o student herda o problema.

**Proteção Existente**: `GenerationConfig` com `temperature=0.0` (greedy) e `max_new_tokens=128`.

### 2.5 Estatísticas com Poucas Seeds
**Arquivo**: [config.py](config.py#L88)
```python
seeds: List[int] = field(default_factory=lambda: [42])
```
**Risco**: Uma única seed não permite intervalos de confiança válidos.

**Recomendação**: Para claims científicos, usar `seeds=[42, 123, 456]` no mínimo.

---

## 3. Ajustes para Google Colab A100

### 3.1 Hiperparâmetros Atualizados em [config.py](config.py)

| Parâmetro | Antes | Depois | Justificativa |
|-----------|-------|--------|---------------|
| `temperature_schedule` | [5.0, 3.0, 1.0] | [2.0, 1.5, 1.0] | Evita amplificação excessiva da KL |
| `alpha_schedule` | [0.7, 0.5, 0.3] | [0.5, 0.4, 0.3] | Mais peso na CE (ground-truth) |
| `learning_rate` | 5e-5 | 3e-5 | Reduzido para estabilidade |
| `epochs` | 4 | 3 | Suficiente com batch maior |
| `batch_size` | 2 | 4 | A100 80GB suporta mais |
| `grad_accum_steps` | 6 | 4 | Effective batch = 16 |
| `clip_grad_norm` | 1.0 | 0.5 | Mais agressivo |
| `dataloader_num_workers` | 0 | 2 | A100 tem CPU suficiente |

### 3.2 Ajustes em [distill.py](distill.py)

| Parâmetro | Antes | Depois | Justificativa |
|-----------|-------|--------|---------------|
| `SLM_KD_MAX_LOGIT_ABS` | 100.0 | 50.0 | Logits mais conservadores |
| `SLM_KD_MIN_SUPERVISED_TOKENS` | 0 (N/A) | 8 | Pula batches degenerados |

---

## 4. Checklist de Validação Pré-Execução

- [ ] Verificar se Drive está montado (para persistência)
- [ ] Confirmar VRAM disponível com `nvidia-smi`
- [ ] Setar variáveis de ambiente se necessário:
  ```bash
  export SLM_KD_DEBUG_NAN=1
  export SLM_KD_CLIP_GRAD_NORM=0.5
  export SLM_KD_MAX_LOGIT_ABS=50
  export SLM_KD_MIN_SUPERVISED_TOKENS=8
  ```
- [ ] Para A100 40GB, reduzir `batch_size` para 2-3

---

## 5. Hierarquia de Modelos

### Configuração Atual (Mantida)
```python
model_hierarchy = {
    "teacher_medium": "Qwen/Qwen2.5-7B-Instruct",
    "student_primary": "Qwen/Qwen2.5-3B-Instruct", 
    "student_small": "Qwen/Qwen2.5-1.5B-Instruct",
}
```

### Justificativa para Manter
- **Mesma família tokenizer**: Qwen2.5 usa o mesmo vocabulário, eliminando mismatch de token IDs
- **Gap suficiente**: 7B → 3B (2.3x) e 7B → 1.5B (4.6x) são ratios típicos para KD
- **Alternativas consideradas**:
  - Qwen 14B → 7B → 3B: Melhor knowledge transfer, mas pode exceder VRAM do A100 40GB
  - Qwen 72B teacher: Requer multi-GPU ou quantização agressiva

### Quando Aumentar o Gap
Se após os ajustes o KD logits ainda apresentar instabilidade:
1. Usar 4-bit quantization no teacher (`load_in_4bit=True`)
2. Considerar teacher maior (14B) com device_map="auto"
3. Aumentar `min_supervised_tokens` para 16

---

## 6. Monitoramento Recomendado

Durante treino, observar:
1. **Loss curves**: Devem decrescer suavemente, sem picos
2. **Grad norm**: Deve estabilizar após warmup
3. **Warnings**: "loss não-finita" deve ser raro (< 1% dos batches)
4. **VRAM**: Monitorar com `nvidia-smi` periodicamente

```python
# Exemplo de monitoramento inline
import os
os.environ["SLM_KD_LOG_EVERY"] = "20"  # Log a cada 20 batches
os.environ["SLM_KD_DEBUG_NAN"] = "1"   # Verbose NaN warnings
```
