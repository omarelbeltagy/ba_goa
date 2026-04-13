# =============================================================================
# benchmarks/loader.py — Dataset loading with the same sampling as the paper
# =============================================================================
# Paper ref: Section 4.1 (p.7) and Appendix C (p.17–18)
#
# Benchmark details from the paper (Appendix C):
#   MMLU:     57 categories, 50 samples each (stratified)  → 2850 total
#   MMLU-Pro: 14 categories, 150 samples each (stratified) → 2100 total
#   GPQA:     full gpqa_diamond split (~198 questions)
#   MATH:     500-problem random subset
#   HumanEval: full set (164 problems)
#   MedMCQA:  full test split
# =============================================================================

import random
from datasets import load_dataset

# Reproducible sampling — paper does not state a seed, we fix one.
SEED = 42


def _sample_stratified(dataset, category_col: str, n_per_category: int) -> list[dict]:
    """Draw n_per_category examples per unique value of category_col."""
    rng = random.Random(SEED)
    by_category: dict[str, list] = {}
    for item in dataset:
        cat = item[category_col]
        by_category.setdefault(cat, []).append(item)

    samples = []
    for cat, items in sorted(by_category.items()):
        rng.shuffle(items)
        samples.extend(items[:n_per_category])
    return samples


def load_mmlu(n_per_category: int = 50) -> list[dict]:
    """Load MMLU with 50 stratified samples per subject.

    Paper ref: Appendix C (p.17–18) —
      "stratified sampling with 50 samples per subject"
    HuggingFace dataset: cais/mmlu, config 'all', split 'test'.

    Returns list of dicts with keys: question, choices, answer, subject.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    samples = _sample_stratified(ds, "subject", n_per_category)
    result = []
    for item in samples:
        result.append({
            "question": item["question"],
            "choices":  item["choices"],      # list of 4 strings
            "answer":   item["answer"],        # int 0–3
            "subject":  item["subject"],
            "benchmark": "mmlu",
        })
    return result


def load_mmlu_pro(n_per_category: int = 150) -> list[dict]:
    """Load MMLU-Pro with 150 stratified samples per category.

    Paper ref: Appendix C (p.18) —
      "stratified sampling with 150 samples per category"
    HuggingFace dataset: TIGER-Lab/MMLU-Pro, split 'test'.

    Returns list of dicts with keys: question, options, answer, category.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = _sample_stratified(ds, "category", n_per_category)
    result = []
    for item in samples:
        result.append({
            "question":  item["question"],
            "choices":   item["options"],      # list of up to 10 strings
            "answer":    item["answer"],        # letter string e.g. "A"
            "category":  item["category"],
            "benchmark": "mmlu_pro",
        })
    return result


def load_gpqa() -> list[dict]:
    """Load full GPQA diamond split.

    Paper ref: Appendix C (p.18) — full set, gpqa_diamond configuration.
    HuggingFace dataset: Idavidrein/gpqa, config 'gpqa_diamond'.

    Returns list of dicts with keys: question, choices, answer.
    """
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    result = []
    for item in ds:
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        # Shuffle so correct answer isn't always index 0
        rng = random.Random(SEED + hash(item["Question"]) % 10000)
        rng.shuffle(choices)
        correct_idx = choices.index(item["Correct Answer"])
        correct_letter = "ABCD"[correct_idx]
        result.append({
            "question":  item["Question"],
            "choices":   choices,
            "answer":    correct_letter,
            "benchmark": "gpqa",
        })
    return result


def load_math(n: int = 500) -> list[dict]:
    """Load a 500-problem subset of MATH.

    Paper ref: Appendix C (p.18) — "We used a subset of 500 problems."
    HuggingFace dataset: lighteval/MATH, split 'test'.

    Returns list of dicts with keys: problem, solution, level, type.
    """
    ds = load_dataset("lighteval/MATH", "all", split="test")
    items = list(ds)
    rng = random.Random(SEED)
    rng.shuffle(items)
    result = []
    for item in items[:n]:
        result.append({
            "question":  item["problem"],
            "solution":  item["solution"],
            "level":     item.get("level", ""),
            "type":      item.get("type", ""),
            "benchmark": "math",
        })
    return result


def load_humaneval() -> list[dict]:
    """Load the full HumanEval benchmark (164 problems).

    Paper ref: Appendix C (p.18) — full set.
    HuggingFace dataset: openai_humaneval, split 'test'.

    Returns list of dicts with keys: task_id, prompt, entry_point, test.
    """
    ds = load_dataset("openai_humaneval", split="test")
    result = []
    for item in ds:
        result.append({
            "task_id":     item["task_id"],
            "question":    item["prompt"],       # the function signature + docstring
            "entry_point": item["entry_point"],
            "test":        item["test"],
            "benchmark":   "humaneval",
        })
    return result


def load_medmcqa() -> list[dict]:
    """Load the full MedMCQA test split.

    Paper ref: Appendix C (p.18) — full set.
    HuggingFace dataset: medmcqa, split 'validation' (test labels are public
    only via the validation split).

    Returns list of dicts with keys: question, choices, answer.
    """
    ds = load_dataset("medmcqa", split="validation")
    option_keys = ["opa", "opb", "opc", "opd"]
    result = []
    for item in ds:
        choices = [item[k] for k in option_keys]
        result.append({
            "question":  item["question"],
            "choices":   choices,
            "answer":    item["cop"],    # int 0–3
            "benchmark": "medmcqa",
        })
    return result


LOADERS = {
    "mmlu":      load_mmlu,
    "mmlu_pro":  load_mmlu_pro,
    "gpqa":      load_gpqa,
    "math":      load_math,
    "humaneval": load_humaneval,
    "medmcqa":   load_medmcqa,
}


def load_benchmark(name: str, **kwargs) -> list[dict]:
    """Unified loader.

    Args:
        name: one of "mmlu", "mmlu_pro", "gpqa", "math", "humaneval", "medmcqa"

    Returns:
        List of question dicts, each with at minimum a "question" key and
        a "benchmark" key for downstream format handling.
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {list(LOADERS)}")
    return LOADERS[name](**kwargs)


def format_question_for_prompt(item: dict) -> str:
    """Format a benchmark question into the prompt string sent to agents.

    For multiple-choice benchmarks, append the labelled choices.
    For MATH and HumanEval, the question text is used directly.
    """
    q = item["question"]
    benchmark = item.get("benchmark", "")

    if benchmark in ("mmlu", "mmlu_pro", "gpqa", "medmcqa"):
        choices = item["choices"]
        labels = "ABCDEFGHIJ"
        choice_lines = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
        return f"{q}\n\n{choice_lines}\n\nAnswer with the letter only."

    if benchmark == "math":
        return f"{q}\n\nSolve step by step and give the final answer."

    if benchmark == "humaneval":
        return f"Complete the following Python function:\n\n{q}"

    return q
