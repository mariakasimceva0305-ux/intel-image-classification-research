from pathlib import Path
import argparse, json, time
import torch
from PIL import Image
from torchvision import transforms
from train import make_model, SIZE


def load(checkpoint):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model(blob["regime"], pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize(144),
            transforms.CenterCrop(blob["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, tf, blob["classes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, tf, classes = load(args.checkpoint)
    with Image.open(args.image) as image:
        x = tf(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probs = model(x).softmax(-1)[0]
    top = probs.topk(3)
    print(
        json.dumps(
            [
                {"class": classes[i], "probability": float(p)}
                for p, i in zip(top.values, top.indices)
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
