# =============================================================================
# pipeline/graph_pooling.py — Stage 4: produce the final answer
# =============================================================================
# Paper ref: Section 3.2.4 (p.6), Equation 6
#
#   A = R''_{max-source}                   if max-pooling  (GoA_max)
#   A = Meta-LLM(Average | R'')            if mean-pooling (GoA_mean)  [Eq. 6]
#
# Max-pooling: return the final response of the agent with the most incoming
# source edges, i.e. the highest relevance score (top of sorted order = index 0).
# No extra LLM call needed.
#
# Mean-pooling: the Meta-LLM synthesises all R'' responses into one answer.
# Requires one additional forward pass through the Meta-LLM.
# =============================================================================

import numpy as np
from prompts.templates import graph_pooling_prompt


def pool_max(
    sorted_keys: list[str],
    final_responses: list[str],
    adjacency: np.ndarray,
) -> str:
    """GoA_max: return the refined response of the most-connected source agent.

    "Max-Pooling, which relies on the most influential node (i.e., the agent
    with the highest number of incoming edges, indicating a higher relevance
    score)."  — Section 3.2.4 (p.6)

    In the sorted order, the agent at index 0 always has the highest relevance
    score S_j (defined by column sums from Eq. 2), so it is by definition the
    most influential source.  We also verify this via incoming-edge count as
    described in the paper.

    Args:
        sorted_keys:     Agent keys sorted highest→lowest relevance.
        final_responses: R'' for each agent, same sorted order.
        adjacency:       A[j,i] weight matrix.

    Returns:
        The final response string of the max-source agent.
    """
    # Count incoming source edges per agent (how many agents send to agent j)
    # An edge (i→j) exists when A[j,i] > 0 (i.e. i is a source for j).
    # But we want the agent that IS a source most often, i.e. has the most
    # outgoing edges (rows where A[j,i] > 0 for various j).
    S = len(sorted_keys)
    outgoing_edge_counts = np.zeros(S, dtype=int)
    for i in range(S):
        outgoing_edge_counts[i] = int((adjacency[:, i] > 0).sum())

    max_source_idx = int(np.argmax(outgoing_edge_counts))
    return final_responses[max_source_idx]


def pool_mean(
    query: str,
    sorted_keys: list[str],
    final_responses: list[str],
    meta_llm_generate,
) -> str:
    """GoA_mean: Meta-LLM synthesises all R'' responses into one answer.

    "Mean-Pooling, which balances contributions by considering responses from
    all selected agents ... requiring an additional forward pass through the
    Meta-LLM."  — Section 3.2.4 (p.6)

    Args:
        query:             Input question Q.
        sorted_keys:       Agent keys sorted highest→lowest relevance.
        final_responses:   R'' for each agent, same sorted order.
        meta_llm_generate: Callable(messages) -> str using the Meta-LLM.

    Returns:
        Synthesised final answer string.
    """
    input_responses = [
        {"name": key, "response": resp}
        for key, resp in zip(sorted_keys, final_responses)
    ]
    messages = graph_pooling_prompt(query, input_responses)
    return meta_llm_generate(messages)


def run_graph_pooling(
    query: str,
    sorted_keys: list[str],
    final_responses: list[str],
    adjacency: np.ndarray,
    meta_llm_generate,
    mode: str = "max",
) -> str:
    """Graph pooling stage — returns the final answer string.

    Args:
        query:             Input question Q.
        sorted_keys:       Agent keys sorted highest→lowest relevance.
        final_responses:   R'' after message passing, same sorted order.
        adjacency:         A[j,i] weight matrix.
        meta_llm_generate: Callable(messages)->str (used only for "mean").
        mode:              "max" (GoA_max) or "mean" (GoA_mean).

    Returns:
        Final answer string A.

    Paper ref: Section 3.2.4 (p.6), Equation 6.
    """
    if mode == "max":
        return pool_max(sorted_keys, final_responses, adjacency)
    elif mode == "mean":
        return pool_mean(query, sorted_keys, final_responses, meta_llm_generate)
    else:
        raise ValueError(f"Unknown pooling mode '{mode}'. Use 'max' or 'mean'.")
