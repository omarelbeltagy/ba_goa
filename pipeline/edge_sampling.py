# =============================================================================
# pipeline/edge_sampling.py — Stage 2: build the weighted adjacency matrix
# =============================================================================
# Paper ref: Section 3.2.2 (p.5–6), Equations 2 and 3
#
#   S_j = Σ_{i≠j} Score_{i→j}                                      [Eq. 2]
#
#   A_{ji} = S_i / Σ_{k ∈ N_j} S_k,  where N_j = {i | (i→j) ∈ E} [Eq. 3]
#
# Each selected agent scores all other agents' initial responses.
# Scores from each agent sum to 1.0 (normalised within that agent's vote).
# The relevance score S_j is the column sum — how much total score agent j
# received from its peers.  Agents with S_j < τ are pruned from the graph.
# The adjacency matrix A_{ji} weights the influence of source i on target j.
# =============================================================================

import re
import numpy as np
from config import TAU
from prompts.templates import edge_sampling_prompt


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_scores(raw_output: str, other_models: list[str]) -> list[float]:
    """Extract per-agent scores from the model's raw text output.

    The paper asks for output in the format:
      'ModelA': 0.6, 'ModelB': 0.3, 'ModelC': 0.1
    (Appendix B, p.16 — example_str format)

    We try name-keyed parsing first, then fall back to positional floats.
    Either way we renormalise to sum exactly to 1.0.

    Args:
        raw_output:   Generated text from an agent's scoring call.
        other_models: Ordered list of the names being scored.

    Returns:
        List of floats, length == len(other_models), summing to 1.0.
    """
    n = len(other_models)
    scores = [0.0] * n

    # Strategy 1: look for 'ModelName': value pairs
    for i, name in enumerate(other_models):
        pattern = rf"['\"]?{re.escape(name)}['\"]?\s*[:\s]\s*([\d.]+)"
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            scores[i] = float(match.group(1))

    # Strategy 2: if strategy 1 found nothing, extract all floats in order
    if sum(scores) == 0.0:
        floats = [float(x) for x in re.findall(r"\d+\.?\d*", raw_output)]
        # Prefer floats in [0,1] that could be scores
        candidates = [f for f in floats if 0.0 <= f <= 1.0]
        for i in range(min(n, len(candidates))):
            scores[i] = candidates[i]

    # Renormalise to exactly 1.0
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    else:
        # Uniform fallback
        scores = [1.0 / n] * n

    return scores


# ── Core computation ──────────────────────────────────────────────────────────

def build_score_matrix(
    query: str,
    agent_keys: list[str],
    responses: list[str],
    generate_fns: dict,
) -> np.ndarray:
    """Collect the full S×S peer-scoring matrix.

    Each agent i scores all other agents j (i≠j).
    score_matrix[i, j] = score that agent i assigns to agent j.
    Diagonal is 0 (agents do not score themselves — Section 3.2.2, p.5:
    "excluding its own to reduce self-bias").

    Args:
        query:       The input question Q.
        agent_keys:  Ordered role keys of the selected agents.
        responses:   Initial responses, same order as agent_keys.
        generate_fns: Dict mapping agent_key → callable(messages) -> str.
                      Each agent must be loaded when called.

    Returns:
        score_matrix: np.ndarray of shape (S, S).
    """
    S = len(agent_keys)
    score_matrix = np.zeros((S, S))

    for i, scorer_key in enumerate(agent_keys):
        # Collect indices and data of all other agents
        other_indices = [j for j in range(S) if j != i]
        other_names   = [agent_keys[j] for j in other_indices]
        other_resps   = [responses[j]  for j in other_indices]

        messages  = edge_sampling_prompt(query, other_names, other_resps)
        raw_output = generate_fns[scorer_key](messages)
        scores     = parse_scores(raw_output, other_names)

        for rank, j in enumerate(other_indices):
            score_matrix[i, j] = scores[rank]

    return score_matrix


def compute_relevance_scores(score_matrix: np.ndarray) -> np.ndarray:
    """Compute per-agent relevance score S_j = column sum.

    S_j = Σ_{i≠j} Score_{i→j}   [Eq. 2, Section 3.2.2]

    Args:
        score_matrix: (S, S) array where score_matrix[i,j] is the score
                      agent i gave to agent j.

    Returns:
        1-D array of length S with relevance scores.
    """
    return score_matrix.sum(axis=0)   # sum over rows for each column j


def prune_and_sort(
    agent_keys: list[str],
    relevance_scores: np.ndarray,
    tau: float = TAU,
) -> tuple[list[str], np.ndarray]:
    """Remove low-relevance agents and sort remaining by score descending.

    Agents with S_j < τ are pruned.
    Source: Section 3.2.2 (p.5) — "Agents with S_j < τ are pruned from the
    communication graph."

    Args:
        agent_keys:        Role keys of currently selected agents.
        relevance_scores:  S_j values, same order as agent_keys.
        tau:               Pruning threshold (default 0.05 from paper).

    Returns:
        (sorted_keys, sorted_scores) with low-relevance agents removed,
        sorted highest-to-lowest relevance (source first).
    """
    kept = [(k, s) for k, s in zip(agent_keys, relevance_scores) if s >= tau]

    if len(kept) == 0:
        # Fallback: keep at least the top agent
        best_idx = int(np.argmax(relevance_scores))
        kept = [(agent_keys[best_idx], relevance_scores[best_idx])]

    kept.sort(key=lambda x: x[1], reverse=True)   # highest S first = sources
    sorted_keys   = [k for k, _ in kept]
    sorted_scores = np.array([s for _, s in kept])
    return sorted_keys, sorted_scores


def build_adjacency_matrix(
    sorted_keys: list[str],
    sorted_scores: np.ndarray,
) -> np.ndarray:
    """Build the weighted directed adjacency matrix A.

    A_{ji} = S_i / Σ_{k ∈ N_j} S_k   [Eq. 3, Section 3.2.2]

    In the sorted order (source = low index, target = high index):
    - Sources (i < j) send messages to targets (j > i).
    - A_{ji} is the weight of source i's message arriving at target j.
    - N_j = all i with i < j (all sources that are "above" j in ranking).

    Args:
        sorted_keys:   Agent keys sorted highest-to-lowest relevance.
        sorted_scores: Corresponding relevance scores S.

    Returns:
        A: np.ndarray of shape (S, S).
           A[j, i] = normalised weight of source i → target j edge.
    """
    S = len(sorted_keys)
    A = np.zeros((S, S))

    for j in range(1, S):                         # targets: indices 1..S-1
        sources  = list(range(j))                 # N_j = all i < j
        denom    = sorted_scores[sources].sum()
        if denom == 0:
            denom = 1.0
        for i in sources:
            A[j, i] = sorted_scores[i] / denom    # Eq. 3

    return A


# ── Public API ────────────────────────────────────────────────────────────────

def run_edge_sampling(
    query: str,
    agent_keys: list[str],
    responses: list[str],
    generate_fns: dict,
    tau: float = TAU,
) -> dict:
    """Full edge-sampling stage: score → prune → sort → build A.

    Args:
        query:        Input question Q.
        agent_keys:   Selected agent keys from node sampling.
        responses:    Each agent's initial response, same order.
        generate_fns: Dict mapping agent_key → callable(messages) -> str.
        tau:          Pruning threshold (paper: 0.05).

    Returns:
        Dict with keys:
          "sorted_keys"    : list[str]       — agents sorted high→low relevance
          "sorted_scores"  : np.ndarray      — their relevance scores S
          "adjacency"      : np.ndarray (S×S)— weighted adjacency matrix A
          "score_matrix"   : np.ndarray (S×S)— raw peer scores (for analysis)

    Paper ref: Section 3.2.2 (p.5–6), Equations 2 and 3.
    """
    score_matrix      = build_score_matrix(query, agent_keys, responses, generate_fns)
    relevance_scores  = compute_relevance_scores(score_matrix)
    sorted_keys, sorted_scores = prune_and_sort(agent_keys, relevance_scores, tau)

    # Rebuild score matrix in sorted order for A computation
    A = build_adjacency_matrix(sorted_keys, sorted_scores)

    return {
        "sorted_keys":   sorted_keys,
        "sorted_scores": sorted_scores,
        "adjacency":     A,
        "score_matrix":  score_matrix,
    }
