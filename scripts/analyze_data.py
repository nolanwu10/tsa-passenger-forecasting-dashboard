from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.analysis import write_markdown_report


if __name__ == "__main__":
    output_path = ROOT / "docs" / "data_readiness_report.md"
    write_markdown_report(output_path)
    print(f"Wrote {output_path}")

