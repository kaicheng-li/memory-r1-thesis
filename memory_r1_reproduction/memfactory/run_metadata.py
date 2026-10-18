"""Record the data, policy settings and adapter used by a local experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from memfactory.retrieval import retrieval_config

PROMPT_CONTRACT = "baseline_87f13c5_prompts_per_question_grpo_v3"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_metadata(args, write=True) -> dict:
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "sentence-transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    adapter = getattr(args, "answer_adapter", None)
    adapter_files = {}
    if adapter:
        for path in sorted(Path(adapter).glob("*")):
            if path.is_file() and (path.suffix in {".safetensors", ".bin"} or path.name == "adapter_config.json"):
                adapter_files[path.name] = file_sha256(path)
    manager_adapter_files = {}
    if getattr(args, "manager_adapter", None):
        for path in sorted(Path(args.manager_adapter).glob("*")):
            if path.is_file() and (path.suffix in {".safetensors", ".bin"} or path.name == "adapter_config.json"):
                manager_adapter_files[path.name] = file_sha256(path)
    payload = {
        "manager_adapter_sha256": manager_adapter_files,
        "retrieval": retrieval_config(),
        "reward_definition": "baseline_lora_unique_overlap_f1_v1",
        "prompt_contract": PROMPT_CONTRACT,
        "data_sha256": file_sha256(args.data_path),
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "commit": commit,
        "dirty_worktree": dirty,
        "packages": packages,
        "answer_adapter_sha256": adapter_files,
        "validation_sha256": (
            file_sha256(getattr(args, "validation_timelines", None) or args.validation_data_path)
            if (getattr(args, "validation_timelines", None) or getattr(args, "validation_data_path", None)) else None
        ),
    }
    if not write:
        return payload
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
