"""Put workspace/detail_lora/ and the repo root on sys.path, as the scripts do when run directly."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
