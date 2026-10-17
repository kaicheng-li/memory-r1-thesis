"""Trusted-local resume at completed tuple boundaries, including all sampling RNGs."""
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def resume_contract(manifest):
    ignored = {"resume", "output_dir", "group_log", "epochs", "checkpoint_every"}
    return {"config": {k: v for k, v in manifest["config"].items()
                       if k not in ignored and not k.startswith("_")},
            **{k: manifest[k] for k in ("data_sha256", "validation_sha256", "answer_adapter_sha256",
                                        "manager_adapter_sha256", "retrieval", "reward_definition", "prompt_contract")}}


def save_training_checkpoint(manager, optimizer, directory, epoch, next_tuple, best, contract):
    directory = Path(directory)
    # Never overwrite a complete checkpoint with a partially written adapter.
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    temporary = directory.with_name(directory.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(temporary)
    manager.tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "rng": capture_rng(),
                "epoch": epoch, "next_tuple": next_tuple, "best": best, "contract": contract},
               temporary / "training_state.pt")
    temporary.rename(directory)
    pointer = directory.parent / "latest_checkpoint.json.tmp"
    pointer.write_text(json.dumps({"path": str(directory.resolve())}), encoding="utf-8")
    pointer.replace(directory.parent / "latest_checkpoint.json")


def read_training_state(directory, contract):
    # Python/numpy RNG objects require weights_only=False: trusted checkpoints only.
    state = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["contract"] != contract:
        raise ValueError("Resume data/model/reward/retrieval configuration differs from checkpoint")
    return state
