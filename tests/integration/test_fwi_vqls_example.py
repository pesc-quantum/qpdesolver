import subprocess
import sys
from pathlib import Path


def test_example_11_runs_successfully():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "examples" / "11_fwi_vqls_model1.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--test-mode"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Script failed with output:\n{res.stdout}\n{res.stderr}"
    assert "FWI VQLS on model_1 completed successfully" in res.stdout

