# =============================================================================
# agents/model_cards.py — Pre-extracted model card summaries
# =============================================================================
# The paper extracts these summaries from HuggingFace READMEs using the
# Meta-LLM at setup time (Appendix B, p.13–14).  They are static per run
# (same for every query), so we cache them here rather than re-extracting
# on every question.
#
# Format: each card follows the 4-field structure from the extraction prompt
# (Appendix B, p.14):  Domain, Task Specialization, Parameter Size,
# Special Features.
#
# You can regenerate these by calling extract_model_card() below if you want
# to verify against the live HuggingFace README.
# =============================================================================

from config import AGENT_KEYS, AGENT_MODELS

# Pre-extracted summaries.
# Source: derived from official HuggingFace model pages for each model.
# The paper's Table 1 footnote (p.7) identifies all six models exactly.
PRECOMPUTED_CARDS: dict[str, str] = {
    "general": (
        "- Domain: General-purpose\n"
        "- Task Specialization: General question answering, instruction following, "
        "logical reasoning, and knowledge retrieval across diverse domains including "
        "mathematics, science, humanities, and coding.\n"
        "- Parameter Size: 7B\n"
        "- Special Features: Trained on a diverse mixture covering mathematics, "
        "coding, reasoning, and multilingual data; strong instruction-following "
        "capability; serves as the Meta-LLM in GoA."
    ),
    "code": (
        "- Domain: Software Engineering / Code\n"
        "- Task Specialization: Code generation, code reasoning, code debugging, "
        "code completion, and code fixing across multiple programming languages.\n"
        "- Parameter Size: 7B\n"
        "- Special Features: Qwen2.5 architecture fine-tuned specifically on "
        "large-scale code corpora; optimized for programming tasks including "
        "competitive coding and software development."
    ),
    "math": (
        "- Domain: Mathematics\n"
        "- Task Specialization: Mathematical reasoning, symbolic computation, "
        "algebraic problem solving, calculus, number theory, and competition-level "
        "mathematics.\n"
        "- Parameter Size: 7B\n"
        "- Special Features: Developed by Mistral AI specifically for mathematical "
        "reasoning; fine-tuned on high-quality mathematical datasets including "
        "competition problems and proof-based tasks."
    ),
    "biomedical": (
        "- Domain: Biomedical / Healthcare\n"
        "- Task Specialization: Biomedical question answering, clinical decision "
        "support, medical knowledge retrieval, disease diagnosis reasoning, and "
        "clinical text understanding.\n"
        "- Parameter Size: 8B\n"
        "- Special Features: Fine-tuned from Llama-3-8B on curated biomedical and "
        "clinical datasets; specialized for medical professional exams and "
        "evidence-based medicine."
    ),
    "finance": (
        "- Domain: Finance\n"
        "- Task Specialization: Financial analysis, investment reasoning, economic "
        "concepts, financial question answering, market analysis, and accounting "
        "knowledge.\n"
        "- Parameter Size: 8B\n"
        "- Special Features: Fine-tuned from Llama-3-8B on financial domain data "
        "covering markets, corporate finance, and macroeconomics; specialized for "
        "financial professional tasks."
    ),
    "legal": (
        "- Domain: Law / Legal\n"
        "- Task Specialization: Legal question answering, statutory interpretation, "
        "case law reasoning, contract analysis, and legal document understanding.\n"
        "- Parameter Size: 7B\n"
        "- Special Features: Fine-tuned on a large corpus of legal texts, statutes, "
        "and case law; instruction-tuned for legal professional tasks including bar "
        "exam-style questions."
    ),
}


def format_model_descriptions(agent_keys: list[str] = None) -> str:
    """Format all model cards into a numbered description block for prompts.

    This is the string passed as `model_descriptions` in the node-sampling
    prompt (Appendix B, p.15).

    Args:
        agent_keys: subset of keys to include; defaults to all six.

    Returns:
        Multi-line string with each model numbered 0..N-1.
    """
    if agent_keys is None:
        agent_keys = AGENT_KEYS

    lines = []
    for idx, key in enumerate(agent_keys):
        card = PRECOMPUTED_CARDS[key]
        model_id = AGENT_MODELS[key]
        lines.append(f"[Model {idx}] ({model_id})\n{card}")
    return "\n\n".join(lines)


def extract_model_card(readme_content: str, generate_fn) -> str:
    """Extract a model card summary live using the Meta-LLM.

    Use this to regenerate PRECOMPUTED_CARDS if you want to verify against the
    live HuggingFace README content.

    Args:
        readme_content: Raw README text fetched from HuggingFace.
        generate_fn:    Callable(messages) -> str — the Meta-LLM generate fn.

    Returns:
        Extracted card string in the 4-field bullet format.

    Paper ref: Appendix B, p.13–14 (Model Card Extraction prompts).
    """
    from prompts.templates import model_card_extraction_prompt
    messages = model_card_extraction_prompt(readme_content)
    return generate_fn(messages)
