import os
import subprocess
import sys

def test_train_script_runs_and_creates_artifacts():
    # Run training
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert os.path.exists("artifacts/heart_disease_pipeline.joblib")
    assert os.path.exists("logs/train_run.json")
