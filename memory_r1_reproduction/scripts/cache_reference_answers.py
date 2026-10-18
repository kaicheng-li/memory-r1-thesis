"""Build all causal raw-reference answers and a report; no Manager updates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_manager_reference import main

if __name__ == "__main__":
    if "--reference-only" not in sys.argv:
        sys.argv.append("--reference-only")
    main()
