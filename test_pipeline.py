import pytest
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from train import ROOT, make_model, train_mode
from predict import load


def test_split_hashes_disjoint_and_all_classes_present():
    d = pd.read_csv(ROOT / "results/split_manifest.csv")
    assert d.groupby("sha256").split.nunique().max() == 1
    for split in ["train", "validation", "test"]:
        assert set(d.loc[d.split == split, "label"]) == set(range(6))


def test_frozen_backbone_and_batchnorm_stay_frozen():
    m = make_model("frozen", pretrained=False)
    train_mode(m, "frozen")
    assert all(
        not p.requires_grad for n, p in m.named_parameters() if not n.startswith("fc.")
    )
    before = m.bn1.running_mean.clone()
    m(torch.randn(2, 3, 128, 128))
    assert torch.equal(before, m.bn1.running_mean)


def test_partial_unfreezes_only_last_block_and_head():
    m = make_model("partial", pretrained=False)
    for n, p in m.named_parameters():
        assert p.requires_grad == (n.startswith("layer4.") or n.startswith("fc."))


def test_checkpoint_inference_matches_saved_predictions():
    if not (ROOT / "results/partial_s42.pt").exists() or not (ROOT / "data").exists():
        pytest.skip("Requires local training checkpoint and dataset")
    torch.set_num_threads(4)
    m, tf, classes = load(ROOT / "results/partial_s42.pt")
    d = pd.read_csv(ROOT / "results/partial_s42_test_predictions.csv").head(16)
    xs = []
    for path in d.path:
        with Image.open(ROOT / "data" / path) as im:
            xs.append(tf(im.convert("RGB")))
    with torch.inference_mode():
        pred = m(torch.stack(xs)).argmax(1).tolist()
    assert pred == d.prediction.tolist()
