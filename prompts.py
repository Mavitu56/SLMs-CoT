from __future__ import annotations

from typing import Tuple


PROMPT_VERSION = "cot_prompt_v1"


def granularity_instruction(granularity_level: int) -> str:
    """Return a short instruction string for reasoning granularity.

    Levels:
    - 0: disabled / backward-compatible (no extra instruction)
    - 1..6: increasing detail

    Kept intentionally short to minimize confounds.
    """

    try:
        level = int(granularity_level)
    except Exception:
        level = 0

    if level <= 0:
        return ""

    # Note: keep the base "Let's think step by step." intact; downstream parsing
    # expects that marker, but can tolerate extra lines after it.
    level = max(1, min(6, level))
    if level == 1:
        return "Keep reasoning minimal (1 sentence)."
    if level == 2:
        return "Keep reasoning brief (2-3 short steps)."
    if level == 3:
        return "Use concise step-by-step reasoning."
    if level == 4:
        return "Use detailed step-by-step reasoning."
    if level == 5:
        return "Use very detailed reasoning with intermediate calculations."
    return "Use extremely detailed reasoning and verify the final answer."


def build_cot_prompt(question: str, *, granularity_level: int = 0, post_cot: bool = False) -> Tuple[str, str]:
    """Build the prompt used for teacher CoT generation and student reasoning distillation.

    Returns:
    - prompt: full prompt string
    - prompt_version: stable identifier for cache versioning
    """

    q = (question or "").strip()
    instr = granularity_instruction(granularity_level)

    if post_cot:
        base = f"Q: {q}\nA: Answer with the final answer first, then provide the reasoning."
        if instr:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1; granularity={int(max(1, min(6, int(granularity_level))))}] {instr}"
        else:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1]"
        # Answer-first scaffold.
        prompt = base + "\n### FINAL_ANSWER:\n"
        return prompt, PROMPT_VERSION

    base = f"Q: {q}\nA: Let's think step by step."
    if instr:
        base = base + f"\n[{PROMPT_VERSION}; granularity={int(max(1, min(6, int(granularity_level))))}] {instr}"
    prompt = base + "\n### REASONING:\n"
    return prompt, PROMPT_VERSION
