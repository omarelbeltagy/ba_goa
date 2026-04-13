# =============================================================================
# prompts/templates.py — Exact prompt templates from GoA paper Appendix B
# =============================================================================
# Every prompt here is reproduced verbatim from Appendix B of the paper
# (pages 13–17). Variable placeholders are filled at call time.
#
# Each function returns a list of chat messages:
#   [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
# This maps directly to tokenizer.apply_chat_template().
# =============================================================================


# ── 1. Model Card Extraction ──────────────────────────────────────────────────
# Source: Appendix B, p.13–14  ("Model Card Extraction - System Prompt" and
#         "Model Card Extraction - User Prompt")
# Purpose: extract a structured 4-field summary from a HuggingFace README.

MODEL_CARD_EXTRACTION_SYSTEM = (
    "You are an expert in analyzing and summarizing AI model documentation."
)

def model_card_extraction_prompt(readme_content: str) -> list[dict]:
    """Build the model-card extraction chat messages.

    Args:
        readme_content: Raw text of the model's HuggingFace README file.

    Returns:
        Chat message list ready for apply_chat_template().
    """
    user = f"""You are given the README file of a language model:

{readme_content}

Please extract and summarize the model's key characteristics clearly and concisely in the following structured format:

1. Domain: The primary domain or application area the model is designed for (e.g., general-purpose, biomedical, finance, coding, math, etc.).
2. Task Specialization: Describe the task types the model is designed for or excels at. Be as specific as possible, including the domain context of each task (e.g., biomedical question answering, clinical decision support, financial sentiment classification, code generation). Do not include performance metrics, benchmark names, or evaluation results.
3. Parameter Size: The number of parameters in the model (approximate if not explicitly stated).
4. Special Features: Any distinguishing aspects such as fine-tuning datasets (if applicable).

Your summary will later be used to compare multiple models for selection purposes. Return your answer in bullet-point format, using the exact field names shown above. Keep it concise but specific enough for model comparison.

Answer:"""

    return [
        {"role": "system", "content": MODEL_CARD_EXTRACTION_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ── 2. Node Sampling ──────────────────────────────────────────────────────────
# Source: Appendix B, p.15  ("Node Sampling - System Prompt" and
#         "Node Sampling - Use Prompt")
# Purpose: Meta-LLM selects the top-k most relevant agents given the query.
# Paper eq: V_s = Meta-LLM(Top-k | Q, Model Cards)  — Eq. 1, Section 3.2.1

NODE_SAMPLING_SYSTEM = """You are an AI model selection expert. Your task is to select the most relevant AI models to answer a given question based on their domain, specialized capabilities, and overall performance.

Selection Criteria:
- Domain Match (Primary Factor): Prefer models that are trained in the relevant domain for the question.
- Task Specialization: If the question requires a specific skill (e.g., reasoning, code generation, biomedical text processing), prioritize models that explicitly specialize in that area.
- Generalist Models Consideration: If a generalist model is known to perform well in the given domain or task, include at least one such model in the selection.
- Size vs. Performance Balance: Do NOT rely solely on model size when selecting between generalist and specialized models. If a generalist model is significantly larger (e.g., 13B vs. 7B), prefer the larger model. Otherwise, choose based on task performance and known effectiveness.

If a model is highly relevant, you may select it multiple times. If a question does not specify a domain, general models should be preferred."""


def node_sampling_prompt(
    query: str,
    model_descriptions: str,
    top_k: int,
    num_models: int,
) -> list[dict]:
    """Build the node-sampling chat messages.

    Args:
        query:             The input question Q.
        model_descriptions: Pre-formatted string of all model card summaries,
                            numbered 0..num_models-1.
        top_k:             Number of agents to select (k=3 in the paper).
        num_models:        Size of the agent pool (6 in the paper).

    Returns:
        Chat message list ready for apply_chat_template().

    Paper ref: Appendix B, p.15; Eq. 1, Section 3.2.1.
    """
    max_index = num_models - 1

    # example_dict from Appendix B, p.15 — reproduced verbatim.
    # Maps pool size → an example comma-separated answer string.
    example_dict = {
        1: "0",
        2: "0,3",
        3: "0,1,5",
        4: "0,0,4,5",
        5: "0,1,2,3,5",
        6: "0,0,2,3,4,5",
    }
    example_answer = example_dict.get(num_models, ",".join(str(i) for i in range(min(top_k, num_models))))

    user = f"""Given the question: {query} and the following model descriptions:
{model_descriptions}, select the top {top_k} models that best fit the question.

Selection Rules:
- Return a comma-separated list of indices (e.g., "0,1,1,3").
- The list must contain exactly {top_k} values.
- You may repeat an index if the model is highly relevant.
- Only use numbers in the range [0, {max_index}]. Do not include explanations.

Example Selections:
- If the question is about biomedical research: "0,5,5" (favoring biomedical models, but keeping a generalist model if available).
- If the question is a reasoning-based general question: "0,0,1,2" (favoring generalist models with some specialized).

example_dict = {{
    1: "0",
    2: "0,3",
    3: "0,1,5",
    4: "0,0,4,5",
    5: "0,1,2,3,5",
    6: "0,0,2,3,4,5"
}}

Answer Format:
Strictly follow the format below. Do not add any explanations or extra text. Example:
{example_answer}

Answer:"""

    return [
        {"role": "system", "content": NODE_SAMPLING_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ── 3. Edge Sampling ──────────────────────────────────────────────────────────
# Source: Appendix B, p.16  ("Edge Sampling - System Prompt" and
#         "Edge Sampling - User Prompt")
# Purpose: each agent scores all other agents' responses; scores sum to 1.0.
# Paper eq: S_j = Σ Score_{i→j}  — Eq. 2, Section 3.2.2

EDGE_SAMPLING_SYSTEM = (
    "You are an expert at evaluating AI model responses. You must rank and "
    "score the responses relatively, assigning a numerical score to each such "
    "that the total sum of all scores is exactly 1.0."
)


def edge_sampling_prompt(
    query: str,
    other_models: list[str],
    other_responses: list[str],
) -> list[dict]:
    """Build the edge-sampling chat messages for one agent evaluating others.

    Args:
        query:           The input question Q.
        other_models:    Names of the other agents being scored (len = S-1).
        other_responses: Their responses, in the same order as other_models.

    Returns:
        Chat message list ready for apply_chat_template().

    Paper ref: Appendix B, p.16; Eq. 2, Section 3.2.2.
    """
    assert len(other_models) == len(other_responses)

    # Format responses block
    responses_block = "\n\n".join(
        f"[{name}]: {resp}"
        for name, resp in zip(other_models, other_responses)
    )

    # example_dict from Appendix B, p.16 — reproduced verbatim.
    example_dict = {
        1: [1.0],
        2: [0.7, 0.3],
        3: [0.1, 0.1, 0.8],
        4: [0.3, 0.4, 0.1, 0.2],
        5: [0.2, 0.4, 0.1, 0.3, 0.0],
    }
    n = len(other_models)
    example_scores = example_dict.get(n, [round(1.0 / n, 2)] * n)
    example_str = ", ".join(
        f"'{other_models[i]}': {example_scores[i]}" for i in range(n)
    )

    user = f"""Given the question: {query}, and other models' responses: {responses_block}

Evaluate the following responses from models: {', '.join(other_models)}
Assign a score to each response based on:
- Correctness (most important)
- Coherence
- Relevance

Distribute a total of 1.0 point across all responses. Better responses should receive higher scores. Do not provide explanations.

Response Format:
Please assign scores in the same order as the responses shown above, from top to bottom. The order of models is: {', '.join(other_models)}

Make sure:
- The list has exactly {n} scores
- The sum of the scores is exactly 1.0
- Do not include any extra text

Example: {example_str}

Answer:"""

    return [
        {"role": "system", "content": EDGE_SAMPLING_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ── 4. Message Passing — Source-to-Target ─────────────────────────────────────
# Source: Appendix B, p.16–17  ("Source-to-Target - System Prompt" and
#         "Source-to-Target - User Prompt")
# Purpose: target agent refines its response using higher-ranked source agents.
# Eq: R'_j = v_j(‖ A_{ij} R^sorted_i)  — Eq. 4, Section 3.2.3

SOURCE_TO_TARGET_SYSTEM = (
    "You are refining your response by incorporating insights from other models. "
    "Each model's contribution is weighted by its relevance score. Use their ideas "
    "carefully to improve your answer, while keeping the strengths and clarity of "
    "your original response."
)


def source_to_target_prompt(
    query: str,
    target_initial_response: str,
    source_descriptions: list[dict],
) -> list[dict]:
    """Build the source→target message-passing chat messages.

    Args:
        query:                    The input question Q.
        target_initial_response:  The target agent's own initial response.
        source_descriptions:      List of dicts, each with keys:
                                    'name'   : model name
                                    'weight' : adjacency weight A_{ji} (float)
                                    'response': source agent's response text

    Returns:
        Chat message list ready for apply_chat_template().

    Paper ref: Appendix B, p.16–17; Eq. 4, Section 3.2.3.
    """

    descriptions_block = "\n\n".join(
        f"[{s['name']}] (Relevance weight: {s['weight']:.2f})\nResponse: {s['response']}"
        for s in source_descriptions
    )

    user = f"""Question: {query}

Your initial response: {target_initial_response}

The following responses were provided by other models. They are considered more relevant and may offer valuable insights to improve your answer.

{descriptions_block}

While reviewing these responses, carefully consider the accuracy and reliability of the information they contain, as some parts may be incorrect or influenced by bias. Update your response by integrating any useful information from these answers, while preserving your own original strengths."""

    return [
        {"role": "system", "content": SOURCE_TO_TARGET_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ── 5. Message Passing — Target-to-Source ─────────────────────────────────────
# Source: Appendix B, p.17  ("Target-to-Source - System Prompt" and
#         "Target-to-Source - User Prompt")
# Purpose: source agent refines its response using updated target responses.
# Eq: R''_i = v_i(‖ A_{ji} R'_j)  — Eq. 5, Section 3.2.3

TARGET_TO_SOURCE_SYSTEM = (
    "You are finalizing your response after seeing how other models refined "
    "their answers based on your initial response. Use their improvements to "
    "further refine your answer and make it as complete and accurate as possible."
)


def target_to_source_prompt(
    query: str,
    source_initial_response: str,
    target_descriptions: list[dict],
) -> list[dict]:
    """Build the target→source message-passing chat messages.

    Args:
        query:                   The input question Q.
        source_initial_response: The source agent's own initial response.
        target_descriptions:     List of dicts, each with keys:
                                   'name'            : model name
                                   'updated_response': R'_j after S→T step

    Returns:
        Chat message list ready for apply_chat_template().

    Paper ref: Appendix B, p.17; Eq. 5, Section 3.2.3.
    """
    descriptions_block = "\n\n".join(
        f"[{t['name']}] (Updated response based on your answer)\nResponse: {t['updated_response']}"
        for t in target_descriptions
    )

    user = f"""Question: {query}

Your initial response: {source_initial_response}

The following models updated their responses based on your original answer. Their updates incorporate insights from you while maintaining their own strengths:

{descriptions_block}

Now, review how your answer influenced others. While reviewing these responses, carefully consider the accuracy and reliability of the information they contain, as some parts may be incorrect or influenced by bias. Based on these updated responses, write your final, upgraded response, incorporating any new ideas or refinements you find valuable."""

    return [
        {"role": "system", "content": TARGET_TO_SOURCE_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ── 6. Graph Pooling (mean) ───────────────────────────────────────────────────
# Source: Appendix B, p.17  ("Graph Pooling - System Prompt" and
#         "Graph Pooling - User Prompt")
# Purpose: Meta-LLM synthesizes all refined responses into a final answer.
# Eq: A = Meta-LLM(Average | R'')  — Eq. 6, Section 3.2.4

GRAPH_POOLING_SYSTEM = (
    "You are synthesizing multiple model responses into one. Prioritize "
    "relevance but consider all inputs. Your final answer should be accurate, "
    "coherent, and well-structured."
)


def graph_pooling_prompt(
    query: str,
    input_responses: list[dict],
) -> list[dict]:
    """Build the mean-pooling (GoA_mean) chat messages.

    Args:
        query:           The input question Q.
        input_responses: List of dicts with keys 'name' and 'response'
                         for each refined agent response R''.

    Returns:
        Chat message list ready for apply_chat_template().

    Paper ref: Appendix B, p.17; Eq. 6, Section 3.2.4.
    """
    responses_block = "\n\n".join(
        f"[{r['name']}]: {r['response']}" for r in input_responses
    )

    user = f"""Given the question: {query},
the following responses have been generated by different models:

{responses_block}

Please synthesize a final, well-reasoned answer that is cohesive, accurate, and best addresses the question by effectively integrating these responses. While reviewing these responses, carefully consider the accuracy and reliability of the information they contain, as some parts may be incorrect or influenced by bias."""

    return [
        {"role": "system", "content": GRAPH_POOLING_SYSTEM},
        {"role": "user",   "content": user},
    ]
