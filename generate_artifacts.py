"""Export summary from recorded training results; no hard-coded metrics."""
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parent

def main():
    results = pd.read_csv(ROOT / "results/test_results.csv")
    selected = results.loc[results["run"] == "partial_s42"].iloc[0]
    payload = {"selection": "validation-selected regime in seed-42 comparison", "run": selected["run"], "test_accuracy": float(selected["accuracy"]), "test_macro_f1": float(selected["macro_f1"])}
    destination = ROOT / "artifacts/verified_summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))

if __name__ == "__main__":
    main()
