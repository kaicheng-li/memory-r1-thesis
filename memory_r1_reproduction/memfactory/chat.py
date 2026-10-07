"""Shared chat formatting for Qwen Instruct prompts."""

from __future__ import annotations


def format_chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_chat_prompt(tokenizer, prompt: str, max_length: int):
    return tokenizer(
        format_chat_prompt(tokenizer, prompt),
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
