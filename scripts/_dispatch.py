from pathlib import Path
import runpy


def run(category: str, script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / category / script_name
    runpy.run_path(str(script_path), run_name="__main__")
