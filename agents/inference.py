# =============================================================================
# agents/inference.py — Model loading and text generation
# =============================================================================
# The paper ran all models on a single A6000 GPU loading one model at a time.
# We replicate that sequential load/unload pattern here.
#
# The ONLY thing that changes between development (quantized) and final runs
# (full precision) is the QUANTIZATION constant in config.py.
# =============================================================================

import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from config import QUANTIZATION, MAX_NEW_TOKENS, DO_SAMPLE


def load_model(model_id: str, quantization: str = QUANTIZATION):
    """Load a model and its tokenizer.

    Args:
        model_id:     HuggingFace model identifier.
        quantization: "4bit" | "8bit" | None (full BF16).
                      Paper used full precision (A6000, BF16).
                      "4bit" is the development default for consumer hardware.

    Returns:
        (model, tokenizer) tuple. Model is on GPU (device_map="auto").
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure a pad token exists (some models omit it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization == "4bit":
        # NF4 + double quantization — best quality/size trade-off at 4-bit.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        # Full precision — matches paper's A6000 setup.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Free GPU memory after a model is no longer needed.

    The paper runs models sequentially on one GPU, so we must unload before
    loading the next model.
    """
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Run one forward pass and return the generated text.

    Args:
        model:          Loaded HuggingFace model.
        tokenizer:      Corresponding tokenizer.
        messages:       Chat message list, e.g.:
                          [{"role": "system", "content": "..."},
                           {"role": "user",   "content": "..."}]
        max_new_tokens: Max tokens to generate. Paper does not specify;
                        512 covers all benchmark answer formats.
        do_sample:      False = greedy decoding (deterministic).
                        Paper says "zero-shot CoT" but does not specify
                        temperature; greedy is the reproducible default.

    Returns:
        Generated text with input tokens stripped.
    """
    # apply_chat_template handles all model-specific formatting
    # (Qwen ChatML, Llama special tokens, Mistral [INST], etc.)
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        # Fallback: some older models don't support system role in template;
        # merge system + user into a single user turn.
        merged_content = "\n\n".join(
            m["content"] for m in messages if m["role"] in ("system", "user")
        )
        fallback_messages = [{"role": "user", "content": merged_content}]
        input_ids = tokenizer.apply_chat_template(
            fallback_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt tokens to get only the generated part
    new_token_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


class Agent:
    """Convenience wrapper: load a model once, call generate many times, then
    unload.  Use as a context manager:

        with Agent("Qwen/Qwen2.5-7B-Instruct") as agent:
            response = agent.generate(messages)
    """

    def __init__(self, model_id: str, quantization: str = QUANTIZATION):
        self.model_id = model_id
        self.quantization = quantization
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        self.model, self.tokenizer = load_model(self.model_id, self.quantization)
        return self

    def __exit__(self, *_):
        unload_model(self.model, self.tokenizer)
        self.model = None
        self.tokenizer = None

    def generate(self, messages: list[dict], **kwargs) -> str:
        return generate(self.model, self.tokenizer, messages, **kwargs)
