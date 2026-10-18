"""Neutral decoding controls shared by training, replay and evaluation."""
from contextlib import contextmanager
from copy import deepcopy

import torch


@contextmanager
def inference_mode(model):
    was_training = model.training
    checkpointing = getattr(model, "is_gradient_checkpointing", False)
    config = getattr(model, "config", None)
    use_cache = getattr(config, "use_cache", None)
    model.eval()
    if checkpointing:
        model.gradient_checkpointing_disable()
    if config is not None:
        config.use_cache = True
    try:
        with torch.no_grad():
            yield
    finally:
        if config is not None:
            config.use_cache = use_cache
        if checkpointing:
            model.gradient_checkpointing_enable()
        model.train(was_training)


def generate_policy(model, tokenizer, inputs, max_new_tokens, temperature, num_return_sequences=1):
    if temperature not in (0.0, 1.0):
        raise ValueError("Use temperature=1 for policy sampling or 0 for greedy evaluation")
    values = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  repetition_penalty=1.0, no_repeat_ngram_size=0,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                  use_cache=True, num_return_sequences=num_return_sequences, num_beams=1)
    configured = getattr(model, "generation_config", None)
    if temperature > 0:
        values.update(temperature=1.0, top_k=0, top_p=1.0, typical_p=1.0)
    if configured is not None:
        configured = deepcopy(configured)
        if temperature == 0:
            # Transformers 5 fills None from model.generation_config again.
            # Use explicit validation-neutral defaults. top_k=50 is ignored
            # under do_sample=False; policy sampling below still uses top_k=0.
            for field, value in (("temperature", 1.0), ("top_k", 50),
                                 ("top_p", 1.0), ("typical_p", 1.0),
                                 ("min_p", None), ("epsilon_cutoff", 0.0),
                                 ("eta_cutoff", 0.0)):
                if hasattr(configured, field):
                    setattr(configured, field, value)
        else:
            for field, value in (("min_p", None), ("epsilon_cutoff", 0.0), ("eta_cutoff", 0.0)):
                if hasattr(configured, field):
                    setattr(configured, field, value)
        for field, value in values.items():
            setattr(configured, field, value)
        values = {"generation_config": configured}
    with inference_mode(model):
        return model.generate(**inputs, **values)
