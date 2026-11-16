import time

PROCESS_STARTED = time.monotonic()

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from .cli import main

main(PROCESS_STARTED)
