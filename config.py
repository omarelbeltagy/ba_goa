# =============================================================================
# Central configuration for GoA replication
# =============================================================================
# All hyperparameters are taken directly from the paper. Citations are inline.
# =============================================================================

# ── Model IDs ────────────────────────────────────────────────────────────────
# Source: Table 1 footnote (p.7) — "Single-agent baselines" listing
AGENT_MODELS = {
    "general":    "Qwen/Qwen2.5-7B-Instruct",
    "code":       "Qwen/Qwen2.5-Coder-7B-Instruct",
    "math":       "mistralai/Mathstral-7B-v0.1",
    "biomedical": "ContactDoctor/Bio-Medical-Llama-3-8B",
    "finance":    "instruction-pretrain/finance-Llama3-8B",
    "legal":      "Equall/Saul-7B-Instruct-v1",
}

# The Meta-LLM is the general-purpose model.
# Source: Section 3.2.1 footnote 1 (p.5) —
META_LLM_KEY = "general"
META_LLM_ID  = AGENT_MODELS[META_LLM_KEY]

# Ordered list for consistent indexin
AGENT_KEYS = ["general", "code", "math", "biomedical", "finance", "legal"]

# ── Pipeline hyperparameters ──────────────────────────────────────────────────
# Source: Section 4.1 (p.7) and Table 5 ablation (p.9)
TOP_K = 3        # number of agents selected per query  (Table 5: best k=3)
TAU   = 0.05     # edge pruning threshold               (Table 5: best τ=0.05)
                 # "Agents with S_j < τ are pruned" — Section 3.2.2 (p.5)

# ── Inference settings ───────────────────────────────────────────────────────
# Greedy decoding for reproducibility. The paper says "zero-shot CoT" but does
# not specify temperature; greedy (do_sample=False) is the standard default.
MAX_NEW_TOKENS = 512
DO_SAMPLE      = False

# ── Quantization ─────────────────────────────────────────────────────────────
# The paper used full precision (BF16) on an A6000 GPU.
# Set to "4bit" for development on consumer hardware.
# Change to None for final paper-quality runs.
# Options: "4bit" | "8bit" | None
QUANTIZATION = "4bit"

# ── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
CACHE_DIR   = "cache"   # stores intermediate per-question pipeline outputs