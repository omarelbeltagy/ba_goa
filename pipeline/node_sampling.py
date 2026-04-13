# =============================================================================
# pipeline/node_sampling.py — Stage 1: select the top-k relevant agents
# =============================================================================
# Paper ref: Section 3.2.1 (p.5), Equation 1
#
#   V_s = Meta-LLM(Top-k | Q, Model Cards)                         [Eq. 1]
#
# The Meta-LLM reads the query and all model card summaries, then returns a
# comma-separated list of exactly k agent indices (repetition allowed).
# =============================================================================

import re
from config import AGENT_KEYS, TOP_K
from agents.model_cards import format_model_descriptions
from prompts.templates import node_sampling_prompt


def parse_node_sampling_output(raw_output: str, num_models: int, top_k: int) -> list[int]:
    """Parse the Meta-LLM's comma-separated index string into a list of ints.

    The paper specifies returning exactly top_k indices in [0, num_models-1].
    We extract all digit tokens, clamp to valid range, and pad/truncate to
    exactly top_k entries if the model deviated from the format.

    Args:
        raw_output: Raw generated text, e.g. "0,1,5" or "Answer: 0, 1, 5"
        num_models: Size of agent pool (6 in the paper).
        top_k:      Required list length (3 in the paper).

    Returns:
        List of exactly top_k integer indices.
    """
    # Extract all integers from the output
    found = [int(x) for x in re.findall(r"\d+", raw_output)]

    # Keep only valid indices
    valid = [x for x in found if 0 <= x < num_models]

    if len(valid) == 0:
        # Fallback: default to general model (index 0) repeated top_k times.
        # The paper notes the general model is always a safe choice.
        return [0] * top_k

    # Pad with the most frequent valid index if too short, truncate if too long.
    while len(valid) < top_k:
        valid.append(valid[-1])
    return valid[:top_k]


def run_node_sampling(
    query: str,
    meta_llm_generate,
    agent_keys: list[str] = None,
    top_k: int = TOP_K,
) -> list[str]:
    """Select the top-k most relevant agents for the given query.

    Args:
        query:             The input question Q.
        meta_llm_generate: Callable(messages) -> str using the Meta-LLM.
        agent_keys:        Ordered list of agent role keys; defaults to all 6.
        top_k:             Number of agents to select (paper default: 3).

    Returns:
        List of top_k agent role keys, e.g. ["biomedical", "general", "general"].
        Repetition is allowed and used by the paper (same agent selected twice
        means it runs twice independently — Appendix B, p.15 example).

    Paper ref: Section 3.2.1 (p.5), Equation 1; Appendix B, p.15.
    """
    if agent_keys is None:
        agent_keys = AGENT_KEYS

    model_descriptions = format_model_descriptions(agent_keys)
    messages = node_sampling_prompt(query, model_descriptions, top_k, len(agent_keys))
    raw_output = meta_llm_generate(messages)

    indices = parse_node_sampling_output(raw_output, len(agent_keys), top_k)
    return [agent_keys[i] for i in indices]
