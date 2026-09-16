"""Training and evaluation entry point for HREC with Llama2-7B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrec.__main__ import main

if __name__ == "__main__":
    main(expected_family="llama")
