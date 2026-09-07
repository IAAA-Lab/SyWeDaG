import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = REPO_ROOT / "examples" / "run_offline_demo.py"
OUTPUT_FILE = REPO_ROOT / "examples" / "output" / "generated_hourly_2021.csv"


def test_offline_demo_runs_and_produces_expected_output():
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert OUTPUT_FILE.exists()

    with open(OUTPUT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 365 * 24
    assert set(rows[0].keys()) == {
        "datetime", "temperature", "precipitation", "wind_speed",
        "wind_direction", "humidity", "pressure",
    }

    OUTPUT_FILE.unlink()
