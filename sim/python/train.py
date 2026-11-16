"""Run from any working directory with the repository's existing .venv Python."""
import time

PROCESS_STARTED = time.monotonic()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# Distilled run: specs are repository-relative, not invocation-directory-relative.
import os
os.chdir(ROOT)

from sim_rl.cli import main

if __name__ == "__main__":
    main(PROCESS_STARTED)
