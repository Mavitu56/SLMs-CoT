# =====================================================================
# CÉLULA E — Avaliação de Acurácia (AUTO-CONTIDA, BATCHED, ~10-15 min/ckpt)
#
# Copie e cole TODO este conteúdo dentro da Célula E no Google Colab.
#
# RECURSOS:
#   - 100% auto-contida (sem dependência de cache de importação do Python)
#   - Batched generation (batch_size=16) com left-padding → 10x mais rápido
#   - Atenção SDPA nativa do PyTorch em bf16
#   - max_new_tokens=384 (tempo de geração otimizado)
#   - Fallback automático para imagens do HuggingFace se necessário
#   - Logs em tempo real a CADA batch no início com ETA e velocidade
#   - Salva incrementalmente no Drive a cada checkpoint concluído
# =====================================================================
import os
import sys
import glob
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

REPO_DIR = "/content/SLMs-CoT"
DRIVE_ROOT = "/content/drive/MyDrive/SLM_ScienceQA_Multimodal"

print("█" * 70)
print("  CÉLULA E: AVALIAÇÃO DE ACURÁCIA (BATCHED, ~10-15 min/ckpt)")
print("█" * 70)

# --- 1. Constantes e Helpers de Avaliação ---
ANSWER_LETTERS = "ABCDE"
HASH_MARKER = "####"
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers science questions. "
    "Think step by step, explain your reasoning clearly, "
    "then provide your final answer after ####."
)

def answer_index_to_letter(index: int) -> str:
    if 0 <= index < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[index]
    return "A"

def format_choices(choices: List[str]) -> str:
    return "\n".join(f"({ANSWER_LETTERS[i]}) {c}" for i, c in enumerate(choices))

def build_user_message(question: str, choices: List[str], image: Optional[Any] = None, hint: str = "") -> dict:
    prompt_text = f"Question: {question}\n\nChoices:\n{format_choices(choices)}"
    if hint and hint.strip():
        prompt_text = f"Hint: {hint.strip()}\n\n{prompt_text}"
    content = []
    if image is not None and isinstance(image, Image.Image):
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt_text})
    return {"role": "user", "content": content}

def extract_scienceqa_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"[Aa]nswer:\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{\s*([A-Ea-e])\s*\}", text)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-Ea-e])\b", text)
    if m:
        return m[-1].upper()
    return None

def _prepare_record(r: Dict[str, Any], processor: AutoProcessor) -> Optional[Tuple[str, Optional[list], str, bool, str]]:
    question = r.get("question", "")
    choices = r.get("choices", [])
    if not question or not choices:
        return None

    if "answer_gold_letter" in r:
        gold_letter = r["answer_gold_letter"].strip().upper()
    elif "answer_gold" in r:
        gold_letter = answer_index_to_letter(r["answer_gold"])
    elif "answer" in r:
        ans_val = r["answer"]
        gold_letter = answer_index_to_letter(ans_val) if isinstance(ans_val, int) else str(ans_val).upper()
    else:
        return None

    image = r.get("image", None)
    if image is None and r.get("image_path") and os.path.isfile(r["image_path"]):
        try:
            image = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            image = None

    has_image = (image is not None and isinstance(image, Image.Image)) or r.get("has_image", False)
    hint = r.get("hint", "") or ""
    subject = r.get("subject", "unknown")

    user_msg = build_user_message(question, choices, image, hint)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_msg]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return text, image_inputs, gold_letter, has_image, subject

@torch.inference_mode()
def evaluate_model_batched(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    records: List[Dict[str, Any]],
    batch_size: int = 48,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()

    total = 0
    correct = 0
    separator_count = 0
    total_with_img = 0
    correct_with_img = 0
    total_no_img = 0
    correct_no_img = 0
    subject_stats: Dict[str, Dict[str, int]] = {}

    # Configurar stop tokens explícitos (<|im_end|> e eos) para não gerar além do fim da resposta
    eos_token_ids = [processor.tokenizer.eos_token_id]
    try:
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_token_ids:
            eos_token_ids.append(im_end_id)
    except Exception:
        pass

    t_start = time.time()
    n_batches = (len(records) + batch_size - 1) // batch_size
    vram_alloc = torch.cuda.memory_allocated(device) / (1024**3)
    vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"  VRAM inicial: {vram_alloc:.1f}GB / {vram_total:.1f}GB", flush=True)
    print(f"  Avaliando {len(records)} exemplos em {n_batches} batches (bs={batch_size}, max_tokens={max_new_tokens})...\n", flush=True)

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    try:
        for batch_idx in range(0, len(records), batch_size):
            batch_records = records[batch_idx:batch_idx + batch_size]
            batch_texts = []
            batch_images = []
            batch_meta = []

            for r in batch_records:
                res = _prepare_record(r, processor)
                if res is None:
                    continue
                text, image_inputs, gold_letter, has_image, subject = res
                batch_texts.append(text)
                if image_inputs:
                    batch_images.extend(image_inputs)
                batch_meta.append((gold_letter, has_image, subject))

            if not batch_texts:
                continue

            inputs = processor(
                text=batch_texts,
                images=batch_images if batch_images else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            prompt_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=eos_token_ids,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

            for i, (gold_letter, has_image, subject) in enumerate(batch_meta):
                generated_tokens = output_ids[i, prompt_len:]
                completion = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                pred_letter = extract_scienceqa_answer(completion)
                is_correct = (pred_letter == gold_letter) if pred_letter else False
                has_sep = HASH_MARKER in completion

                total += 1
                if is_correct:
                    correct += 1
                if has_sep:
                    separator_count += 1

                if has_image:
                    total_with_img += 1
                    if is_correct:
                        correct_with_img += 1
                else:
                    total_no_img += 1
                    if is_correct:
                        correct_no_img += 1

                sub_entry = subject_stats.setdefault(subject, {"total": 0, "correct": 0})
                sub_entry["total"] += 1
                if is_correct:
                    sub_entry["correct"] += 1

            del inputs, output_ids
            # NÃO chamamos torch.cuda.empty_cache() dentro do loop para evitar sincronização CPU/GPU forçada!

            # Feedback de progresso
            elapsed = time.time() - t_start
            current_batch = batch_idx // batch_size + 1
            s_per_ex = elapsed / max(total, 1)
            remaining = (len(records) - total) * s_per_ex
            eta_str = f"{remaining/60:.1f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"
            acc_pct = (correct / max(total, 1)) * 100.0
            pct = (total / len(records)) * 100.0
            vram_now = torch.cuda.memory_allocated(device) / (1024**3)

            if current_batch <= 5 or current_batch % 5 == 0 or current_batch == n_batches:
                acc_img_pct = (correct_with_img / max(total_with_img, 1)) * 100.0 if total_with_img > 0 else 0.0
                acc_noimg_pct = (correct_no_img / max(total_no_img, 1)) * 100.0 if total_no_img > 0 else 0.0
                print(
                    f"  [Batch {current_batch:>3d}/{n_batches} ({pct:>5.1f}%)] "
                    f"avaliados={total:>4d}/{len(records)} | "
                    f"acc={acc_pct:.1f}% (img={acc_img_pct:.1f}%, texto={acc_noimg_pct:.1f}%) | "
                    f"VRAM={vram_now:.1f}GB | vel={s_per_ex:.2f}s/ex | ETA={eta_str}",
                    flush=True,
                )
    finally:
        processor.tokenizer.padding_side = original_padding_side

    elapsed_total = time.time() - t_start
    print(f"\n  ✓ Avaliação finalizada em {elapsed_total/60:.1f} min ({total} exemplos)", flush=True)

    return {
        "n_total": total,
        "n_correct": correct,
        "accuracy_overall": correct / max(total, 1),
        "n_with_image": total_with_img,
        "accuracy_with_image": correct_with_img / max(total_with_img, 1) if total_with_img > 0 else None,
        "n_without_image": total_no_img,
        "accuracy_without_image": correct_no_img / max(total_no_img, 1) if total_no_img > 0 else None,
        "separator_adherence": separator_count / max(total, 1),
    }

# --- 2. Localizar Checkpoints no Drive ---
print("\n[Passo 1/3] Buscando checkpoints no Google Drive...", flush=True)
checkpoints = sorted(glob.glob(f"{DRIVE_ROOT}/*/checkpoints/final"))
print(f"  Encontrados {len(checkpoints)} checkpoints:")
for ckpt in checkpoints:
    norm_p = os.path.normpath(ckpt)
    r_name = os.path.basename(os.path.dirname(os.path.dirname(norm_p)))
    print(f"  • {r_name}")

if len(checkpoints) == 0:
    print(f"  ⚠️ Nenhum checkpoint encontrado em {DRIVE_ROOT}/*/checkpoints/final")
else:
    # --- 3. Carregar Dados de Teste ---
    cot_jsonl = f"{REPO_DIR}/data/scienceqa_cot_qwen25_vl_7b.jsonl"
    if not os.path.isfile(cot_jsonl):
        cot_jsonl = f"{DRIVE_ROOT}/data/scienceqa_cot_qwen25_vl_7b.jsonl"

    print(f"\n[Passo 2/3] Carregando dataset de teste de {cot_jsonl}...", flush=True)
    records = []
    with open(cot_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)
    print(f"  ✓ {len(records)} exemplos de teste carregados")

    # Fallback automático de imagens do HuggingFace se necessário
    need_hf = False
    for r in records[:50]:
        if r.get("has_image") and (not r.get("image_path") or not os.path.isfile(r.get("image_path", ""))):
            need_hf = True
            break

    if need_hf:
        print("  ℹ️ Vinculando imagens do HuggingFace (derek-thomas/ScienceQA)...", flush=True)
        try:
            import datasets
            ds = datasets.load_dataset("derek-thomas/ScienceQA", split="test")
            linked = 0
            for r in records:
                idx = r.get("idx")
                if idx is not None and idx < len(ds) and r.get("has_image"):
                    r["image"] = ds[idx].get("image")
                    linked += 1
            print(f"  ✓ {linked} imagens vinculadas com sucesso!")
        except Exception as e:
            print(f"  ⚠️ Aviso: não foi possível vincular imagens do HF: {e}")

    print("  Carregando processador Qwen2.5-VL-3B-Instruct...", flush=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)

    # Filtrar pendentes (pula checkpoints já avaliados)
    pending = []
    for ckpt in checkpoints:
        norm_p = os.path.normpath(ckpt)
        r_name = os.path.basename(os.path.dirname(os.path.dirname(norm_p)))
        out_json = f"{DRIVE_ROOT}/results/eval_acc_{r_name}.json"
        if os.path.isfile(out_json) and os.path.getsize(out_json) > 10:
            print(f"  ⏩ PULADO: {r_name} (já avaliado em {out_json})")
        else:
            pending.append((ckpt, r_name, out_json))

    BATCH_SIZE = 48        # 48 a 64 utiliza ~25-35 GB de VRAM na A100 e acelera 3x a 4x
    MAX_NEW_TOKENS = 256   # 90%+ dos raciocínios têm < 250 tokens; evita alucinações longas

    # --- 4. Executar Avaliações ---
    print(f"\n[Passo 3/3] Avaliando {len(pending)} checkpoints pendentes (batch_size={BATCH_SIZE})...", flush=True)
    eval_start = time.time()

    for idx, (ckpt, run_name, out_json) in enumerate(pending, 1):
        print(f"\n{'═' * 70}")
        print(f"▶ [{idx}/{len(pending)}] Avaliando: {run_name}")
        print(f"  Origem: {ckpt}")
        print(f"{'═' * 70}", flush=True)

        t0 = time.time()

        print("  Carregando checkpoint em bf16 + SDPA...", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        results = evaluate_model_batched(
            model=model,
            processor=processor,
            records=records,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        elapsed = (time.time() - t0) / 60.0
        acc = results["accuracy_overall"] * 100
        acc_img = results.get("accuracy_with_image")
        acc_noimg = results.get("accuracy_without_image")

        print(f"\n  📊 RESULTADOS: {run_name}")
        print(f"     Acurácia Geral:      {acc:.2f}% ({results['n_correct']}/{results['n_total']})")
        if acc_img is not None:
            print(f"     Com Imagem (Visual): {acc_img*100:.2f}% ({results['n_with_image']} amostras)")
        if acc_noimg is not None:
            print(f"     Sem Imagem (Texto):  {acc_noimg*100:.2f}% ({results['n_without_image']} amostras)")
        print(f"     Tempo do modelo:     {elapsed:.1f} minutos")
        print(f"     Salvo no Drive em:   {out_json}", flush=True)

        del model
        torch.cuda.empty_cache()

    # --- Resumo Consolidado ---
    print(f"\n{'═' * 70}")
    print("📊 RESUMO GERAL DE ACURÁCIA — SCIENCEQA MULTIMODAL")
    print(f"{'═' * 70}")
    print(f"{'Experimento':38s} | {'Geral':>8s} | {'Com Img':>8s} | {'Sem Img':>8s}")
    print("-" * 70)

    result_files = sorted(glob.glob(f"{DRIVE_ROOT}/results/eval_acc_*.json"))
    for rf in result_files:
        name = os.path.basename(rf).replace("eval_acc_", "").replace(".json", "")
        try:
            with open(rf, "r", encoding="utf-8") as fh:
                r = json.load(fh)
            acc = r.get("accuracy_overall", 0) * 100
            acc_i = r.get("accuracy_with_image")
            acc_ni = r.get("accuracy_without_image")
            s_img = f"{acc_i*100:.1f}%" if acc_i is not None else "N/A"
            s_noimg = f"{acc_ni*100:.1f}%" if acc_ni is not None else "N/A"
            print(f"{name:38s} | {acc:>7.1f}% | {s_img:>8s} | {s_noimg:>8s}")
        except Exception as e:
            print(f"{name:38s} | Erro: {e}")

    print(f"{'═' * 70}")
    total_time = (time.time() - eval_start) / 60.0
    print(f"✓ CÉLULA E CONCLUÍDA em {total_time:.0f} minutos!")
    print(f"{'═' * 70}\n")
