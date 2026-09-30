import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
runpy.run_path(str(Path(__file__).parent / "scripts" / "benchmark.py"), run_name="__main__")
