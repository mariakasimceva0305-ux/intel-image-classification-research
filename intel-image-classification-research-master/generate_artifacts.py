from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


def write_results_summary() -> None:
    rows = [
        {
            "experiment": "resnet18_full_finetune",
            "model": "ResNet18 (full fine-tuning)",
            "val_accuracy": 0.933048,
            "val_macro_f1": 0.933917,
            "test_accuracy": 0.927333,
            "test_macro_f1": 0.928300,
            "comment": "Лучшая модель",
            "source": "notebook logs: sections 17/19/20",
        },
        {
            "experiment": "resnet18_partial_unfreeze",
            "model": "ResNet18 (partial unfreeze)",
            "val_accuracy": 0.923077,
            "val_macro_f1": 0.923813,
            "test_accuracy": None,
            "test_macro_f1": None,
            "comment": "Есть val-метрики, test не выгружен в логе",
            "source": "notebook logs: section 17",
        },
        {
            "experiment": "resnet18_frozen",
            "model": "ResNet18 (frozen backbone)",
            "val_accuracy": 0.891738,
            "val_macro_f1": 0.893521,
            "test_accuracy": None,
            "test_macro_f1": None,
            "comment": "Есть val-метрики, test не выгружен в логе",
            "source": "notebook logs: section 17",
        },
        {
            "experiment": "deep_cnn_bn_aug",
            "model": "DeepCNN + BatchNorm + Augmentation",
            "val_accuracy": 0.773029,
            "val_macro_f1": 0.774108,
            "test_accuracy": None,
            "test_macro_f1": None,
            "comment": "Есть val-метрики, test не выгружен в логе",
            "source": "notebook logs: section 17",
        },
        {
            "experiment": "small_cnn_baseline",
            "model": "SmallCNN",
            "val_accuracy": 0.646724,
            "val_macro_f1": 0.645374,
            "test_accuracy": None,
            "test_macro_f1": None,
            "comment": "Есть val-метрики, test не выгружен в логе",
            "source": "notebook logs: section 17",
        },
    ]
    df = pd.DataFrame(rows)
    (ARTIFACTS_DIR / "results_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def write_classification_report() -> None:
    rows = [
        {"label": "buildings", "precision": 0.890086, "recall": 0.945080, "f1-score": 0.916759, "support": 437},
        {"label": "forest", "precision": 0.987395, "recall": 0.991561, "f1-score": 0.989474, "support": 474},
        {"label": "glacier", "precision": 0.899441, "recall": 0.873418, "f1-score": 0.886239, "support": 553},
        {"label": "mountain", "precision": 0.901961, "recall": 0.876190, "f1-score": 0.888889, "support": 525},
        {"label": "sea", "precision": 0.945076, "recall": 0.978431, "f1-score": 0.961464, "support": 510},
        {"label": "street", "precision": 0.942268, "recall": 0.912176, "f1-score": 0.926978, "support": 501},
        {"label": "accuracy", "precision": 0.927333, "recall": 0.927333, "f1-score": 0.927333, "support": 0.927333},
        {"label": "macro avg", "precision": 0.927705, "recall": 0.929476, "f1-score": 0.928300, "support": 3000},
        {"label": "weighted avg", "precision": 0.927326, "recall": 0.927333, "f1-score": 0.927051, "support": 3000},
    ]
    df = pd.DataFrame(rows)
    (ARTIFACTS_DIR / "best_model_classification_report.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figures are already copied to artifacts/figures from supplied notebook outputs.
    write_results_summary()
    write_classification_report()

    print("Generated:")
    print(f"- {ARTIFACTS_DIR / 'results_summary.csv'}")
    print(f"- {ARTIFACTS_DIR / 'best_model_classification_report.csv'}")


if __name__ == "__main__":
    main()
