"""Identical chat formatting and truncation for sampling and policy scoring."""

DEFAULT_MAX_PROMPT_TOKENS = 4096


def tokenize_chat_prompt(tokenizer, prompt: str, max_length: int = DEFAULT_MAX_PROMPT_TOKENS):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.truncation_side = "left"
    return tokenizer(
        formatted,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
