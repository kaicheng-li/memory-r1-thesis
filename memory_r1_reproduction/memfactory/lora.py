"""Shared QLoRA model loading for Memory-R1."""

from __future__ import annotations

import torch


def load_lora_model(
    base_model_path: str,
    device: str,
    adapter_path: str | None = None,
    trainable: bool = False,
):
    if not device.startswith("cuda"):
        raise RuntimeError("QLoRA requires a CUDA device.")

    if not trainable and adapter_path is None:
        return load_reference_model(base_model_path, device)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_index = 0 if device == "cuda" else device
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": device_index},
    )
    model = prepare_model_for_kbit_training(model)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    else:
        model = get_peft_model(model, lora_config)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable and "lora_" in name)
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train(trainable)
    if trainable:
        model.print_trainable_parameters()
    return model


def load_reference_model(base_model_path: str, device: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_index = 0 if device == "cuda" else device
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": device_index},
    )
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
