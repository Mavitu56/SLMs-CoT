from __future__ import annotations

from typing import Optional, Tuple


PROMPT_VERSION = "cot_prompt_v2"


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


def _one_shot_prefix(prefix: Optional[str]) -> str:
    p = (prefix or "").strip("\n")
    return (p + "\n\n") if p else ""


def build_teacher_cot_prompt(
    question: str,
    *,
    granularity_level: int = 0,
    post_cot: bool = False,
    one_shot_prefix: Optional[str] = None,
    gold_answer: Optional[str] = None,
    post_cot_gold_rationale: bool = False,
) -> Tuple[str, str]:
    """Prompt for TEACHER CoT/rationale generation.

    - Default: matches the student scaffold (`build_cot_prompt`).
    - Post-CoT gold rationale: provides the gold answer and asks only for a
      justification (teacher remains frozen).
    """

    q = (question or "").strip()
    instr = granularity_instruction(granularity_level)
    prefix = _one_shot_prefix(one_shot_prefix)

    if post_cot and post_cot_gold_rationale:
        a = (gold_answer or "").strip()
        base = f"Q: {q}\nA: The correct final answer is: {a}\nExplain why this answer is correct."
        if instr:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1; gold_rationale=1; granularity={int(max(1, min(6, int(granularity_level))))}] {instr}"
        else:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1; gold_rationale=1]"
        prompt = prefix + base + "\n### REASONING:\n"
        return prompt, PROMPT_VERSION

    prompt, _ = build_cot_prompt(
        q,
        granularity_level=int(granularity_level or 0),
        post_cot=bool(post_cot),
        one_shot_prefix=one_shot_prefix,
    )
    return prompt, PROMPT_VERSION


def build_cot_prompt(
    question: str,
    *,
    granularity_level: int = 0,
    post_cot: bool = False,
    one_shot_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Prompt used for STUDENT distillation.

    Returns:
    - prompt: full prompt string
    - prompt_version: stable identifier for prompt/version tracking
    """

    q = (question or "").strip()
    instr = granularity_instruction(granularity_level)
    prefix = _one_shot_prefix(one_shot_prefix)

    if post_cot:
        base = f"Q: {q}\nA: Answer with the final answer first, then provide the reasoning."
        if instr:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1; granularity={int(max(1, min(6, int(granularity_level))))}] {instr}"
        else:
            base = base + f"\n[{PROMPT_VERSION}; post_cot=1]"
        # Answer-first scaffold.
        prompt = prefix + base + "\n### FINAL_ANSWER:\n"
        return prompt, PROMPT_VERSION

    base = f"Q: {q}\nA: Let's think step by step."
    if instr:
        base = base + f"\n[{PROMPT_VERSION}; granularity={int(max(1, min(6, int(granularity_level))))}] {instr}"
    prompt = prefix + base + "\n### REASONING:\n"
    return prompt, PROMPT_VERSION


def build_one_shot_demo(
    *,
    question: str,
    answer: str,
    reasoning: str,
    post_cot: bool,
) -> str:
    """Short one-shot demo that shows the expected output format (student-side)."""

    q = (question or "").strip()
    a = (answer or "").strip()
    r = (reasoning or "").strip()

    if post_cot:
        return (
            "Example:\n"
            + f"Q: {q}\n"
            + "A: Answer with the final answer first, then provide the reasoning.\n"
            + "### FINAL_ANSWER:\n"
            + f"{a}\n"
            + "### REASONING:\n"
            + f"{r}\n"
        )

    return (
        "Example:\n"
        + f"Q: {q}\n"
        + "A: Let's think step by step.\n"
        + "### REASONING:\n"
        + f"{r}\n"
        + "### FINAL_ANSWER: "
        + f"{a}\n"
    )


def build_one_shot_teacher_demo_for_post_cot_gold_rationale(
    *,
    question: str,
    answer: str,
    reasoning: str,
) -> str:
    """One-shot demo for teacher prompt when using gold-conditioned Post-CoT rationale."""

    q = (question or "").strip()
    a = (answer or "").strip()
    r = (reasoning or "").strip()

    return (
        "Example:\n"
        + f"Q: {q}\n"
        + f"A: The correct final answer is: {a}\n"
        + "Explain why this answer is correct.\n"
        + "### REASONING:\n"
        + f"{r}\n"
    )


def build_cascod_rationale_prompt(
    question: str,
    *,
    granularity_level: int = 0,
) -> str:
    """CasCoD stage1 prompt: q -> r (no answer target)."""

    prompt, _ = build_cot_prompt(
        question,
        granularity_level=int(granularity_level or 0),
        post_cot=False,
        one_shot_prefix=None,
    )
    return prompt


def build_cascod_answer_prompt(
    question: str,
    rationale: str,
) -> str:
    """CasCoD stage2 prompt: q,r -> a (answer-only target)."""

    q = (question or "").strip()
    r = (rationale or "").strip()

    return f"Q: {q}\n### REASONING:\n{r}\n### FINAL_ANSWER:\n"
