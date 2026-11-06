"""Put the repo root (sarbench/, make_submission.py), workspace/r50fpn/ and workspace/sarmtl/ on sys.path for the tests."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # workspace/r50fpn
sys.path[:0] = [str(HERE.parents[1]), str(HERE), str(HERE.parent / 'sarmtl')]
