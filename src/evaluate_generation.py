"""Generation-based evaluation: ROUGE-L for Dolly, Exact Match for MMLU.

ROUGE-L evaluates open-ended generation quality on the Dolly test set.
MMLU Exact Match evaluates multiple-choice accuracy via logit comparison.

These metrics are more expensive than probabilistic metrics:
- ROUGE-L: ~30 min per checkpoint (generation-based)
- MMLU accuracy: ~5-10 min per checkpoint (forward pass only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

# We import rouge_score lazily to avoid import errors if not installed
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ------------------------------------------------------------------
# ROUGE-L Evaluation for Dolly
# ------------------------------------------------------------------

def _format_dolly_prompt(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Format a Dolly example into a chat-template prompt (without response).
    
    Returns the prompt string ready for generation.
    """
    instruction = example["instruction"]
    context = example.get("context", "")
    
    if context and context.strip():
        prompt_content = f"{instruction}\n\n{context}"
    else:
        prompt_content = instruction
    
    messages = [{"role": "user", "content": prompt_content}]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text


@torch.no_grad()
def evaluate_rouge_l_dolly(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 256,
    max_prompt_length: int = 256,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate ROUGE-L on the Dolly test set via text generation with real batching.
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        The model to evaluate (must be in eval mode).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer matching the model.
    max_new_tokens : int
        Maximum tokens to generate per example.
    max_prompt_length : int
        Maximum tokens for the prompt (truncated if longer).
    batch_size : int
        Batch size for generation. Uses left padding for proper batching.
    show_progress : bool
        Whether to show a progress bar.
        
    Returns
    -------
    dict with keys:
        rouge_l_precision_mean, rouge_l_recall_mean, rouge_l_fmeasure_mean,
        n_examples, n_empty_generations, avg_generation_length
    """
    if not ROUGE_AVAILABLE:
        raise ImportError(
            "rouge_score is not installed. "
            "Install with: pip install rouge-score"
        )
    
    from src.data_dolly import load_dolly
    
    model.eval()
    device = next(model.parameters()).device
    
    # Load raw Dolly test set
    ds = load_dolly(split="test")
    n_examples = len(ds)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Accumulators
    rouge_l_precision_sum = 0.0
    rouge_l_recall_sum = 0.0
    rouge_l_fmeasure_sum = 0.0
    n_empty_generations = 0
    total_generation_tokens = 0
    
    # Configure tokenizer for left padding (required for batch generation)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        # Process in batches
        iterator = tqdm(
            range(0, n_examples, batch_size),
            desc="ROUGE-L eval",
            disable=not show_progress,
        )
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_examples)
            batch_examples = [ds[i] for i in range(start_idx, end_idx)]
            
            # Skip examples with empty references
            valid_examples = []
            references = []
            for example in batch_examples:
                reference = example["response"]
                if reference and reference.strip():
                    valid_examples.append(example)
                    references.append(reference)
            
            if not valid_examples:
                continue
            
            # Format prompts
            prompts = [_format_dolly_prompt(ex, tokenizer) for ex in valid_examples]
            
            # Tokenize batch with left padding
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
            ).to(device)
            
            # Generate
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Process each generated sequence
            for i, (reference, output_seq) in enumerate(zip(references, output_ids)):
                # Extract generated tokens (exclude prompt)
                prompt_len = inputs["attention_mask"][i].sum().item()  # Non-pad tokens
                generated_ids = output_seq[inputs["input_ids"].shape[1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Track empty generations
                if not generated_text.strip():
                    n_empty_generations += 1
                    continue
                
                total_generation_tokens += len(generated_ids)
                
                # Compute ROUGE-L
                scores = scorer.score(reference, generated_text)
                rouge_l = scores['rougeL']
                
                rouge_l_precision_sum += rouge_l.precision
                rouge_l_recall_sum += rouge_l.recall
                rouge_l_fmeasure_sum += rouge_l.fmeasure
            
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
    
    # Compute averages
    n_valid = n_examples - n_empty_generations
    if n_valid == 0:
        return {
            "rouge_l_precision_mean": 0.0,
            "rouge_l_recall_mean": 0.0,
            "rouge_l_fmeasure_mean": 0.0,
            "n_examples": n_examples,
            "n_empty_generations": n_empty_generations,
            "avg_generation_length": 0.0,
        }
    
    return {
        "rouge_l_precision_mean": rouge_l_precision_sum / n_valid,
        "rouge_l_recall_mean": rouge_l_recall_sum / n_valid,
        "rouge_l_fmeasure_mean": rouge_l_fmeasure_sum / n_valid,
        "n_examples": n_examples,
        "n_empty_generations": n_empty_generations,
        "avg_generation_length": total_generation_tokens / n_valid,
    }


# ------------------------------------------------------------------
# MMLU Exact Match Evaluation
# ------------------------------------------------------------------

def _get_answer_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, int]:
    """Get token IDs for answer letters A, B, C, D.
    
    Uses the format " A", " B", etc. (with leading space) as this is
    how the answer appears after "Answer:" in the prompt.
    
    Returns
    -------
    dict mapping letter to token ID
    """
    answer_tokens = {}
    for letter in "ABCD":
        # Try with space prefix first (most common case)
        ids_with_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        # Also try without space
        ids_no_space = tokenizer.encode(letter, add_special_tokens=False)
        
        # Use the single-token version if available
        if len(ids_with_space) == 1:
            answer_tokens[letter] = ids_with_space[0]
        elif len(ids_no_space) == 1:
            answer_tokens[letter] = ids_no_space[0]
        else:
            # Fallback: use first token of the with-space version
            answer_tokens[letter] = ids_with_space[0]
    
    return answer_tokens


@torch.no_grad()
def evaluate_mmlu_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    batch_size: int = 8,
    subject: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate exact-match accuracy on MMLU test set.
    
    For each example:
    1. Forward pass to get logits
    2. Extract logits at the last prompt token position
    3. Compare argmax over {A, B, C, D} token logits with correct answer
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        The model to evaluate (must be in eval mode).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer matching the model.
    max_length : int
        Maximum sequence length for tokenization.
    batch_size : int
        Batch size for forward pass.
    subject : str or None
        MMLU subject to evaluate (None = all subjects).
    show_progress : bool
        Whether to show a progress bar.
        
    Returns
    -------
    dict with keys:
        mmlu_accuracy, n_correct, n_total, per_subject_accuracy (if all subjects)
    """
    from src.data_mmlu import load_mmlu, format_mmlu_prompt

    model.eval()
    device = next(model.parameters()).device

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    
    try:
        # Get answer token IDs
        answer_token_ids = _get_answer_token_ids(tokenizer)
        print(f"[mmlu] Answer token IDs: {answer_token_ids}")

        # Load MMLU
        ds = load_mmlu(split="test", subject=subject)
        n_total = len(ds)

        # Accumulators
        n_correct = 0
        per_subject_correct: Dict[str, int] = {}
        per_subject_total: Dict[str, int] = {}

        # Process in batches
        iterator = tqdm(
            range(0, n_total, batch_size),
            desc="MMLU eval",
            disable=not show_progress,
        )

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_total)
            batch_examples = [ds[i] for i in range(start_idx, end_idx)]

            # Format prompts and get correct answers
            prompts = []
            correct_letters = []
            subjects = []

            for ex in batch_examples:
                prompt_text, answer_letter = format_mmlu_prompt(ex)

                # Build chat-formatted prompt (user message only)
                messages = [{"role": "user", "content": prompt_text}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(chat_prompt)
                correct_letters.append(answer_letter)
                subjects.append(ex.get("subject", "unknown"))

            # Tokenize batch
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(device)

            # Forward pass
            outputs = model(**encoded)
            logits = outputs.logits  # [B, L, V]

            # For each example, get logits at last non-pad position
            for i, (correct_letter, subj) in enumerate(zip(correct_letters, subjects)):
                # Find last non-pad position
                attention_mask = encoded["attention_mask"][i]
                last_pos = attention_mask.sum().item() - 1

                # Get logits at last position
                last_logits = logits[i, last_pos]  # [V]

                # Extract logits for A, B, C, D
                answer_logits = torch.tensor([
                    last_logits[answer_token_ids["A"]].item(),
                    last_logits[answer_token_ids["B"]].item(),
                    last_logits[answer_token_ids["C"]].item(),
                    last_logits[answer_token_ids["D"]].item(),
                ])

                # Predict: argmax over the 4 choices
                predicted_idx = answer_logits.argmax().item()
                predicted_letter = "ABCD"[predicted_idx]

                # Check correctness
                is_correct = (predicted_letter == correct_letter)
                if is_correct:
                    n_correct += 1

                # Track per-subject
                if subj not in per_subject_total:
                    per_subject_total[subj] = 0
                    per_subject_correct[subj] = 0
                per_subject_total[subj] += 1
                if is_correct:
                    per_subject_correct[subj] += 1

        # Compute accuracies
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        per_subject_accuracy = {
            subj: per_subject_correct[subj] / per_subject_total[subj]
            for subj in per_subject_total
        }

        return {
            "mmlu_accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject_accuracy": per_subject_accuracy,
        }
    finally:
        tokenizer.padding_side = original_padding_side


# ------------------------------------------------------------------
# Unified evaluation function
# ------------------------------------------------------------------

def evaluate_generation_metrics(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    cfg: Dict[str, Any],
    run_rouge: bool = True,
    run_mmlu: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run all generation-based metrics.
    
    Parameters
    ----------
    model : The model to evaluate.
    tokenizer : Tokenizer matching the model.
    cfg : Config dict.
    run_rouge : Whether to run ROUGE-L evaluation on Dolly.
    run_mmlu : Whether to run MMLU exact match evaluation.
    show_progress : Whether to show progress bars.
    
    Returns
    -------
    dict with all computed metrics
    """
    results: Dict[str, Any] = {}
    
    if run_rouge:
        print("\n[generation] Running ROUGE-L evaluation on Dolly test set...")
        rouge_results = evaluate_rouge_l_dolly(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.get("rouge_max_new_tokens", 256),
            max_prompt_length=cfg.get("rouge_max_prompt_length", 256),
            batch_size=cfg.get("rouge_batch_size", 32),
            show_progress=show_progress,
        )
        results.update(rouge_results)
        print(f"[generation] ROUGE-L F1: {rouge_results['rouge_l_fmeasure_mean']:.4f}")
    
    if run_mmlu:
        print("\n[generation] Running MMLU exact-match evaluation...")
        mmlu_results = evaluate_mmlu_accuracy(
            model=model,
            tokenizer=tokenizer,
            max_length=cfg.get("max_length", 512),
            batch_size=cfg.get("batch_size", 8),
            subject=cfg.get("mmlu_subject", None),
            show_progress=show_progress,
        )
        results.update(mmlu_results)
        print(f"[generation] MMLU Accuracy: {mmlu_results['mmlu_accuracy']:.4f}")
    
    return results
