"""Put the repo root (sarbench/, make_submission.py) and workspace/sarmtl/ on sys.path for the tests."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # workspace/sarmtl
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
