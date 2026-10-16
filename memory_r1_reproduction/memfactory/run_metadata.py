"""Record the data, policy settings and adapter used by a local experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROMPT_CONTRACT = "speaker_grouped_public_memory_v2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_metadata(args) -> dict:
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
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
    payload = {
        "prompt_contract": PROMPT_CONTRACT,
        "data_sha256": file_sha256(args.data_path),
        "config": vars(args),
        "commit": commit,
        "dirty_worktree": dirty,
        "packages": packages,
        "answer_adapter_sha256": adapter_files,
        "validation_sha256": (
            file_sha256(args.validation_timelines)
            if getattr(args, "validation_timelines", None) else None
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
