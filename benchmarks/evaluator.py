# =============================================================================
# benchmarks/evaluator.py — Answer extraction and accuracy scoring
# =============================================================================
# Paper ref: Section 4.1 (p.7) — "All performance is measured using
#   zero-shot CoT in test-time."
# The paper does not describe exact answer-extraction logic; we use standard
# practices for each benchmark type.
# =============================================================================

import re


# ── Multiple-choice answer extraction ────────────────────────────────────────
# Used for MMLU, MMLU-Pro, GPQA, MedMCQA

def extract_mc_answer(text: str, num_choices: int = 4) -> str | None:
    """Extract a single letter answer (A–J) from model output.

    Tries several patterns in priority order:
      1. "Answer: X" or "The answer is X"
      2. "(X)" standalone
      3. Last standalone capital letter in valid range

    Returns None if extraction fails.
    """
    valid = set("ABCDEFGHIJ"[:num_choices])
    text_upper = text.upper()

    # Pattern 1: explicit answer markers
    for pattern in [
        r"(?:ANSWER|THE ANSWER IS|FINAL ANSWER)[:\s]+([A-J])\b",
        r"\b([A-J])\s*(?:IS CORRECT|IS THE ANSWER)",
        r">> FINAL ANSWER:\s*([A-J])\b",
        r"\*\*([A-J])\*\*",       # bold letter
    ]:
        m = re.search(pattern, text_upper)
        if m and m.group(1) in valid:
            return m.group(1)

    # Pattern 2: "(X)" standalone
    m = re.search(r"\(([A-J])\)", text_upper)
    if m and m.group(1) in valid:
        return m.group(1)

    # Pattern 3: last standalone capital letter
    for letter in reversed(re.findall(r"\b([A-J])\b", text_upper)):
        if letter in valid:
            return letter

    return None


def mc_answer_to_index(letter: str | None, num_choices: int = 4) -> int | None:
    """Convert letter answer to zero-based index."""
    if letter is None:
        return None
    idx = "ABCDEFGHIJ".index(letter)
    return idx if idx < num_choices else None


# ── MATH answer extraction ────────────────────────────────────────────────────

def extract_math_answer(text: str) -> str:
    """Extract the final boxed answer from a MATH response.

    The paper evaluates MATH accuracy; standard practice is to look for
    \\boxed{...} in the response.
    """
    # Look for \\boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # Fallback: last number or expression in the text
    numbers = re.findall(r"-?\d+(?:[./]\d+)?", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def normalize_math(s: str) -> str:
    """Light normalisation for MATH answer comparison."""
    s = s.replace(" ", "").replace(",", "")
    # Remove leading zeros in integers
    s = re.sub(r"\b0+(\d)", r"\1", s)
    return s.lower()


# ── HumanEval scoring ─────────────────────────────────────────────────────────

def extract_code(text: str, entry_point: str) -> str:
    """Extract the Python function from the model's response.

    Tries to find a ```python ... ``` block first, then falls back to the
    first function definition in the text.
    """
    # Code fence
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Look for the function definition
    m = re.search(rf"def {re.escape(entry_point)}\s*\(.*", text, re.DOTALL)
    if m:
        return text[m.start():].strip()

    return text.strip()


# ── Per-benchmark scoring ─────────────────────────────────────────────────────

def score_mmlu(item: dict, prediction: str) -> bool:
    """item['answer'] is int 0–3; convert to letter, compare to prediction."""
    correct_letter = "ABCD"[item["answer"]]
    pred_letter    = extract_mc_answer(prediction, num_choices=4)
    return pred_letter == correct_letter


def score_mmlu_pro(item: dict, prediction: str) -> bool:
    """item['answer'] is already a letter string e.g. 'A'."""
    correct_letter = item["answer"].upper()
    num_choices    = len(item["choices"])
    pred_letter    = extract_mc_answer(prediction, num_choices=num_choices)
    return pred_letter == correct_letter


def score_gpqa(item: dict, prediction: str) -> bool:
    """item['answer'] is a letter string e.g. 'B'."""
    correct_letter = item["answer"].upper()
    pred_letter    = extract_mc_answer(prediction, num_choices=4)
    return pred_letter == correct_letter


def score_math(item: dict, prediction: str) -> bool:
    """Exact-match after normalisation."""
    pred   = normalize_math(extract_math_answer(prediction))
    # Extract answer from solution if needed
    sol = normalize_math(extract_math_answer(item["solution"]))
    return pred == sol


def score_medmcqa(item: dict, prediction: str) -> bool:
    """item['answer'] is int 0–3."""
    correct_letter = "ABCD"[item["answer"]]
    pred_letter    = extract_mc_answer(prediction, num_choices=4)
    return pred_letter == correct_letter


SCORERS = {
    "mmlu":      score_mmlu,
    "mmlu_pro":  score_mmlu_pro,
    "gpqa":      score_gpqa,
    "math":      score_math,
    "humaneval": None,           # handled separately (requires code execution)
    "medmcqa":   score_medmcqa,
}


def score(item: dict, prediction: str) -> bool:
    """Score a single prediction against the ground truth.

    Args:
        item:       Benchmark item dict (output of loader functions).
        prediction: Model's raw generated text.

    Returns:
        True if the prediction is correct.
    """
    benchmark = item.get("benchmark", "")
    scorer = SCORERS.get(benchmark)
    if scorer is None:
        raise NotImplementedError(
            f"Scoring for '{benchmark}' requires code execution. "
            "Run evaluation with the 'human-eval' package separately."
        )
    return scorer(item, prediction)


def compute_accuracy(results: list[dict]) -> float:
    """Compute accuracy from a list of result dicts with 'correct' bool key."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results) * 100.0
