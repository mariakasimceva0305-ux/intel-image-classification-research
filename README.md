# Scene classification: transfer learning under a fixed CPU budget

Six scene classes from the Intel Image Classification dataset. Four training regimes are compared: a small CNN trained from scratch and ImageNet-pretrained ResNet18 with frozen, partially trainable, or fully trainable weights.

## Protocol

The experiment uses 4,902 training images, 2,101 validation images and 2,997 test images. The remaining 7,003 images are unused to keep CPU training bounded. Exact byte duplicates are removed; hashes with conflicting labels are excluded, and test hashes are excluded from the training pool. This does not rule out visually similar near-duplicates.

All regimes use 128-pixel inputs, four epochs, batch size 32 and the same split. Validation macro-F1 selects each checkpoint. All six predefined training runs finish before test evaluation. Frozen layers retain frozen batch-normalization statistics. Seed 42 compares the four regimes; seeds 43 and 44 additionally measure full fine-tuning variability on the same data split.

## Results

Test macro-F1: CNN 0.7100; frozen ResNet18 0.9013; partial fine-tuning 0.9172. Full fine-tuning scores 0.9204, 0.9147 and 0.9115 across three seeds. The mean and sample standard deviation are recorded in `results/summary.json`.

Partial fine-tuning is the validation-selected regime in the primary seed-42 comparison. The better test score of full fine-tuning at seed 42 is not used to change that selection. The experiment demonstrates transfer-learning gains under this budget, not a globally optimal architecture or a production CV system. These are new measurements from 10 October 2026, not reproductions of the earlier notebook's reported scores.

![Results](results/comparison.png)

## Reproduce

Use Python 3.12 with the versions in `environment.json`. Install PyTorch 2.10.0 and torchvision 0.25.0 from the official CPU wheel index. Obtain the dataset from https://www.kaggle.com/datasets/puneet6060/intel-image-classification and place the labelled folders under `data/` as listed in `results/split_manifest.csv`.

```text
python train.py
python -m pytest -q test_pipeline.py
python predict.py --checkpoint results/partial_s42.pt --image path/to/image.jpg
```

Training resumes completed runs in the output folder; use a fresh output directory for independent reproduction (`python train.py --help`). Inference loads saved weights without downloading pretrained weights. The tests verify split separation, frozen batch-normalization, the partial-training parameter scope and checkpoint prediction agreement. Checkpoints and original images remain local and are not bundled for publication.

Data origin, split audit, per-image predictions, confusion matrices, learning curves and per-class reports are retained in `results/`. ImageNet initialization is provided by torchvision's ResNet18 IMAGENET1K_V1 weights. Dataset licensing and redistribution conditions must be checked before redistributing images.
