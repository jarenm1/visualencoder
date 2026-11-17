"""Record a trained policy for offline first-/third-person browser replay."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from sim_rl.replay import main

if __name__ == "__main__":
    main()
