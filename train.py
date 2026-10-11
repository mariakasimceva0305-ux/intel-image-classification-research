"""Scene classification: fixed split, four training regimes, held-out evaluation."""

from pathlib import Path
import argparse, copy, hashlib, json, random, time, os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
SIZE = 128
EPOCHS = 4
BATCH = 32


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 6),
        )

    def forward(self, x):
        return self.net(x)


class Images(Dataset):
    def __init__(self, rows, training):
        self.rows = rows.reset_index(drop=True)
        spatial = (
            [
                transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
            if training
            else [transforms.Resize(144), transforms.CenterCrop(SIZE)]
        )
        self.tf = transforms.Compose(
            spatial
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        with Image.open(ROOT / "data" / r.path) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(r.label)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_model(regime, pretrained=True):
    if regime == "small_cnn":
        return SmallCNN()
    m = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    m.fc = nn.Linear(m.fc.in_features, 6)
    if regime in ("frozen", "partial"):
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
        if regime == "partial":
            for p in m.layer4.parameters():
                p.requires_grad = True
    return m


def train_mode(m, regime):
    m.train()
    if regime in ("frozen", "partial"):
        m.eval()
        m.fc.train()
        if regime == "partial":
            m.layer4.train()


def evaluate(m, loader):
    m.eval()
    ys = []
    ps = []
    loss = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = m(x)
            loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            ys.extend(y.tolist())
            ps.extend(logits.argmax(1).tolist())
    return (
        dict(
            loss=loss / len(ys),
            accuracy=accuracy_score(ys, ps),
            macro_f1=f1_score(
                ys, ps, labels=range(6), average="macro", zero_division=0
            ),
        ),
        ys,
        ps,
    )


def prepare(out):
    manifest = out / "split_manifest.csv"
    if manifest.exists():
        return pd.read_csv(manifest)
    rows = []
    for p in sorted((ROOT / "data").rglob("*.jpg")):
        if p.parent.name not in CLASSES:
            continue
        parts = p.parts
        split = "test" if "seg_test" in parts else "pool"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(
            dict(
                path=p.relative_to(ROOT / "data").as_posix(),
                label=CLASSES.index(p.parent.name),
                sha256=digest,
                origin=split,
            )
        )
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("No labelled images. Run prepare_intel.py first.")
    ambiguous = set(d.groupby("sha256").label.nunique().loc[lambda x: x > 1].index)
    d = d[~d.sha256.isin(ambiguous)]
    test = d[d.origin == "test"].drop_duplicates("sha256").copy()
    pool = (
        d[(d.origin == "pool") & ~d.sha256.isin(test.sha256)]
        .drop_duplicates("sha256")
        .copy()
    )
    chosen, unused = train_test_split(
        pool, test_size=0.5, random_state=42, stratify=pool.label
    )
    train, val = train_test_split(
        chosen, test_size=0.3, random_state=42, stratify=chosen.label
    )
    records = []
    for part, name in [
        (train, "train"),
        (val, "validation"),
        (test, "test"),
        (unused, "unused"),
    ]:
        records.append(part.assign(split=name))
    final = pd.concat(records, ignore_index=True)
    assert final.groupby("sha256").split.nunique().max() == 1
    final.to_csv(manifest, index=False)
    (out / "data_audit.json").write_text(
        json.dumps(
            dict(
                raw_images=len(rows),
                ambiguous_hashes=len(ambiguous),
                retained_images=len(final),
                counts=final.split.value_counts().to_dict(),
                exact_duplicate_policy="deduplicate; exclude test hashes from train pool; discard conflicting-label hashes",
            ),
            indent=2,
        )
    )
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = prepare(out)
    protocol = dict(
        dataset="Intel Image Classification",
        classes=CLASSES,
        image_size=SIZE,
        batch_size=BATCH,
        epochs=EPOCHS,
        split_seed=42,
        train_fraction=0.35,
        validation_fraction=0.15,
        regimes=["small_cnn", "frozen", "partial", "full"],
        runs=[
            ["small_cnn", 42],
            ["frozen", 42],
            ["partial", 42],
            ["full", 42],
            ["full", 43],
            ["full", 44],
        ],
        selection="best validation macro-F1 checkpoint; no test-driven tuning",
        notes=[
            "New experiment; different resolution and budget from historical README.",
            "Full fine-tuning seed replications share the same fixed split.",
            "Only exact byte duplicates are removed; near duplicates not exhaustively excluded.",
        ],
    )
    proto = out / "protocol.json"
    if proto.exists() and json.loads(proto.read_text()) != protocol:
        raise RuntimeError("Protocol differs from existing run")
    proto.write_text(json.dumps(protocol, indent=2))
    train = rows[rows.split == "train"]
    val = rows[rows.split == "validation"]
    test = rows[rows.split == "test"]
    vloader = DataLoader(Images(val, False), batch_size=BATCH, num_workers=0)
    for regime, seed in protocol["runs"]:
        name = f"{regime}_s{seed}"
        done = out / f"{name}_training.json"
        if done.exists():
            continue
        seed_all(seed)
        m = make_model(regime)
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            Images(train, True),
            batch_size=BATCH,
            shuffle=True,
            generator=g,
            num_workers=0,
        )
        if regime == "small_cnn":
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            opt = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for n, p in m.named_parameters()
                            if p.requires_grad and not n.startswith("fc.")
                        ],
                        "lr": 1e-4,
                    },
                    {"params": m.fc.parameters(), "lr": 1e-3},
                ],
                weight_decay=1e-4,
            )
        best = -1
        best_epoch = None
        history = []
        start = time.time()
        for epoch in range(EPOCHS):
            train_mode(m, regime)
            loss_sum = 0
            n = 0
            for x, y in loader:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(m(x), y)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(y)
                n += len(y)
            metrics, _, _ = evaluate(m, vloader)
            r = dict(
                epoch=epoch + 1,
                train_loss=loss_sum / n,
                **metrics,
                seconds=time.time() - start,
            )
            history.append(r)
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "regime": regime,
                        "seed": seed,
                        "state_dict": m.state_dict(),
                        "classes": CLASSES,
                        "image_size": SIZE,
                    },
                    out / f"{name}.pt",
                )
            (out / f"{name}_curve.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(run=name, **r)), flush=True)
        done.write_text(
            json.dumps(
                dict(
                    run=name,
                    regime=regime,
                    seed=seed,
                    best_epoch=best_epoch,
                    validation_macro_f1=best,
                    seconds=time.time() - start,
                ),
                indent=2,
            )
        )
    # Test is evaluated only after all predeclared training runs finish.
    tloader = DataLoader(Images(test, False), batch_size=BATCH, num_workers=0)
    results = []
    for regime, seed in protocol["runs"]:
        name = f"{regime}_s{seed}"
        m = make_model(regime, pretrained=False)
        blob = torch.load(out / f"{name}.pt", weights_only=True)
        m.load_state_dict(blob["state_dict"])
        metrics, y, p = evaluate(m, tloader)
        training = json.loads((out / f"{name}_training.json").read_text())
        results.append(dict(**training, **metrics))
        pd.DataFrame({"path": test.path.tolist(), "target": y, "prediction": p}).to_csv(
            out / f"{name}_test_predictions.csv", index=False
        )
        (out / f"{name}_classification_report.json").write_text(
            json.dumps(
                classification_report(
                    y, p, target_names=CLASSES, output_dict=True, zero_division=0
                ),
                indent=2,
            )
        )
        np.save(out / f"{name}_confusion.npy", confusion_matrix(y, p, labels=range(6)))
    pd.DataFrame(results).to_csv(out / "test_results.csv", index=False)
    print("COMPLETE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
