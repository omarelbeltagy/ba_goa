# =============================================================================
# pipeline/message_passing.py — Stage 3: bidirectional message passing
# =============================================================================
# Paper ref: Section 3.2.3 (p.6), Equations 4 and 5
#
#   Source→Target:  R'_j  = v_j( ‖_{i<j} A_{ij} · R^sorted_i )   [Eq. 4]
#   Target→Source:  R''_i = v_i( ‖_{j>i} A_{ji} · R'_j       )   [Eq. 5]
#
# In the sorted ordering (index 0 = highest relevance = source):
#   - Sources are agents with lower sorted index (more relevant).
#   - Targets are agents with higher sorted index (less relevant).
#
# Step 1 (Source→Target): each target j receives and refines its response
#   using the initial responses of all sources i < j, weighted by A[j,i].
#
# Step 2 (Target→Source): each source i refines its response using the
#   UPDATED responses R'_j of all targets j > i, weighted by A[j,i].
#   This lets sources incorporate the consensus formed around their own output.
# =============================================================================

import numpy as np
from prompts.templates import source_to_target_prompt, target_to_source_prompt


def run_source_to_target(
    query: str,
    sorted_keys: list[str],
    sorted_scores: np.ndarray,
    initial_responses: list[str],
    adjacency: np.ndarray,
    generate_fns: dict,
) -> list[str]:
    """Source-to-Target message passing step.

    Each target agent j (j > 0 in sorted order) generates an updated response
    R'_j by seeing the initial responses of all source agents i < j, weighted
    by the adjacency A[j,i].

    Source agents (index 0, or any agent with no sources above them) keep their
    initial response unchanged after this step — only their R'' changes in the
    Target→Source step.

    Args:
        query:             Input question Q.
        sorted_keys:       Agent keys sorted highest→lowest relevance.
        sorted_scores:     Corresponding S_j values (for building descriptions).
        initial_responses: R_i — each agent's initial response, sorted order.
        adjacency:         A[j,i] — weight matrix from edge_sampling.
        generate_fns:      Dict mapping agent_key → callable(messages)->str.

    Returns:
        updated_responses: list[str] of length S.
                           Index 0 (top source) is unchanged (= initial).
                           Indices 1..S-1 are updated responses R'_j.

    Paper ref: Section 3.2.3 (p.6), Equation 4.
    """
    S = len(sorted_keys)
    updated_responses = list(initial_responses)   # copy; index 0 unchanged

    for j in range(1, S):                         # targets only
        source_indices = list(range(j))           # all i < j are sources

        # Build the description list for the prompt.
        # We pass A[j,i] as the "relevance weight" so the LLM can calibrate
        # how much to defer to each source (system prompt says "weighted by
        # its relevance score" — Appendix B, p.16).
        source_descs = [
            {
                "name":     sorted_keys[i],
                "weight":   float(adjacency[j, i]),
                "response": initial_responses[i],
            }
            for i in source_indices
        ]

        messages = source_to_target_prompt(
            query,
            target_initial_response=initial_responses[j],
            source_descriptions=source_descs,
        )
        updated_responses[j] = generate_fns[sorted_keys[j]](messages)

    return updated_responses


def run_target_to_source(
    query: str,
    sorted_keys: list[str],
    initial_responses: list[str],
    updated_responses: list[str],
    adjacency: np.ndarray,
    generate_fns: dict,
) -> list[str]:
    """Target-to-Source message passing step.

    Each source agent i (i < S-1) generates a final refined response R''_i
    by seeing the UPDATED responses R'_j of all targets j > i.

    The final target (index S-1) has no targets below it, so its R'' == R'.

    Args:
        query:             Input question Q.
        sorted_keys:       Agent keys sorted highest→lowest relevance.
        initial_responses: R_i — agents' initial responses (pre-S→T).
        updated_responses: R'_j — agents' responses after the S→T step.
        adjacency:         A[j,i] — weight matrix.
        generate_fns:      Dict mapping agent_key → callable(messages)->str.

    Returns:
        final_responses: list[str] of length S, the R'' values.
                         The last agent (pure target) keeps R' unchanged.

    Paper ref: Section 3.2.3 (p.6), Equation 5.
    """
    S = len(sorted_keys)
    final_responses = list(updated_responses)     # copy; last index unchanged

    for i in range(S - 1):                       # sources (all but last)
        target_indices = list(range(i + 1, S))   # all j > i are targets

        # Build description of each target's updated response.
        # A[j,i] is the weight of source i on target j; here we just list
        # the targets' refined responses so the source can see the "consensus
        # that formed around its own initial answer" (Section 3.2.3, p.6).
        target_descs = [
            {
                "name":             sorted_keys[j],
                "updated_response": updated_responses[j],
            }
            for j in target_indices
        ]

        messages = target_to_source_prompt(
            query,
            source_initial_response=initial_responses[i],
            target_descriptions=target_descs,
        )
        final_responses[i] = generate_fns[sorted_keys[i]](messages)

    return final_responses


def run_message_passing(
    query: str,
    sorted_keys: list[str],
    sorted_scores: np.ndarray,
    initial_responses: list[str],
    adjacency: np.ndarray,
    generate_fns: dict,
) -> dict:
    """Full bidirectional message-passing stage.

    Runs Source→Target then Target→Source, as described in Section 3.2.3.

    Returns:
        Dict with keys:
          "updated_responses" : list[str]  — R'  after S→T step
          "final_responses"   : list[str]  — R'' after T→S step

    Paper ref: Section 3.2.3 (p.6), Equations 4 and 5.
    """
    updated = run_source_to_target(
        query, sorted_keys, sorted_scores,
        initial_responses, adjacency, generate_fns,
    )
    final = run_target_to_source(
        query, sorted_keys,
        initial_responses, updated,
        adjacency, generate_fns,
    )
    return {
        "updated_responses": updated,
        "final_responses":   final,
    }