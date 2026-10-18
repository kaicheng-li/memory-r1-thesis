"""Shared checkpoint/manifest contracts and rank-0 operations."""
import json
from pathlib import Path
import torch.distributed as dist

from memfactory.reference_answer import atomic_json, digest
from memfactory.run_metadata import file_sha256
from memfactory.retrieval import retrieval_config


def main_call(accelerator, function):
    if accelerator.num_processes == 1:
        return function()
    from memfactory.lora import _CPU_SYNC_GROUP
    message = [None]
    if accelerator.is_main_process:
        try:
            message[0] = (True, function())
        except Exception as exc:
            message[0] = (False, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(message, src=0, group=_CPU_SYNC_GROUP)
    if not message[0][0]:
        raise RuntimeError(f"Rank 0: {message[0][1]}")
    return message[0][1]


def run_contract(args, world_size, model_identity, reference_protocol):
    ignored = {"output_dir", "resume_from", "progress_log", "group_log", "save_every_groups",
               "reference_cache_dir", "comparison_log", "reference_only", "device"}
    config = {k:v for k,v in vars(args).items() if k not in ignored and not k.startswith("_")}
    files = {k:file_sha256(getattr(args,k)) for k in
             ("data_path", "validation_states", "validation_timelines", "reference_timelines")}
    root = Path(__file__).resolve().parents[1]
    sources = {p:file_sha256(root/p) for p in (
        "scripts/train_manager_reference.py", "memfactory/reference_advantage.py",
        "memfactory/reference_answer.py", "memfactory/generation.py", "memfactory/manager_env.py",
        "memfactory/reference_runtime.py", "memfactory/training_state.py", "memfactory/lora.py",
        "memfactory/retrieval.py", "scripts/construct_memory_bank.py", "evaluation/manager_validation.py",
        "memfactory/modules/memory_updater.py", "memfactory/modules/memory_retriever.py",
        "evaluation/metrics.py", "memfactory/chat.py")}
    return {"schema":"single_step_reference_v1", "config":config, "world_size":world_size,
            "files":files, "sources":sources, "model":model_identity,
            "reference_protocol_hash":digest(reference_protocol), "retrieval":retrieval_config()}


def prepare_output(args, contract):
    output = Path(args.output_dir)
    if not args.resume_from and (output/"run_manifest.json").exists():
        raise ValueError("Output already contains a run; use --resume-from or a new output directory")
    atomic_json(output/"run_manifest.json", contract)


def selection_record(epoch, adapter, f1, em):
    # Explicit null is a legitimate base-model winner, never a missing selection.
    return {"epoch":epoch, "adapter":str(Path(adapter).resolve()) if adapter else None,
            "path":str(Path(adapter).resolve()) if adapter else None,
            "roll_forward_f1":f1, "roll_forward_em":em}
