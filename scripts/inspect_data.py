from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

