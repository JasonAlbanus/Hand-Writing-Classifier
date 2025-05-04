"""
predict_handwriting_gui.py
--------------------------

Small Tkinter GUI for one-image handwriting recognition (IAM word level).

- Loads a fine-tuned ResNet model once at startup.
- Lets the user pick an image via a file-chooser.
- Shows the image and predicted word in the window.

Usage
-----
$ python predict_handwriting_gui.py --model MODEL.pth --label-map label2idx.json
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ── GUI & imaging ────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageOps

# ── ML stack ─────────────────────────────────────────────────────────────────
import torch
from torchvision import transforms, models


# -----------------------------------------------------------------------------
# 1. Command-line options
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GUI for handwritten word prediction")
    p.add_argument("--model", required=True, type=Path,
                   help="Path to saved model (.dump / .pth / TorchScript)")
    p.add_argument("--label-map", type=Path, default=None,
                   help="(optional) JSON mapping label→index; "
                        "if absent we look inside the checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def extract_idx2label(state: dict) -> Optional[Dict[int, str]]:
    """Return idx→label dict if it exists inside the checkpoint."""
    for key in ("idx2label", "label2idx"):
        if key in state:
            mapping = state[key]
            if isinstance(mapping, dict):
                if key == "label2idx":                # invert
                    mapping = {v: k for k, v in mapping.items()}
                return {int(k): str(v) for k, v in mapping.items()}
    return None
# -----------------------------------------------------------------------------
# 2. Pre-processing pipeline (identical to training)
# -----------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Lambda(lambda img:
        transforms.functional.resize(
            img, size=(128, int(img.width * 128 / img.height)))),
    transforms.Pad((0, 0, 16, 0), fill=255),
    transforms.ToTensor(),   # values in [0,1]
])
# -----------------------------------------------------------------------------
# 3. Helper functions
# -----------------------------------------------------------------------------
def load_label_map(json_path: Path):
    with json_path.open() as fp:
        label2idx = json.load(fp)
    return {int(v): k for k, v in label2idx.items()}

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn, torch

class WordResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.conv1.stride               = (1, 1)
        net.layer4[0].conv1.stride     = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(net.children())[:-1])  # no FC
        self.dropout  = nn.Dropout(0.3)
        self.head     = nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)
    
class WordCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        def block(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2)
            )

        self.features = torch.nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

def load_model(weights: Path, device: str):
    state = torch.load(str(weights), map_location="cpu")
    if "state_dict" in state:                      # lightning-style
        state = {k.replace("model.", "", 1): v
                 for k, v in state["state_dict"].items()}

    # get label map from ckpt if present
    idx2label = extract_idx2label(state)

    # how many output nodes?
    fc_key = [k for k in state if k.endswith(".weight") and state[k].ndim == 2][0]
    num_classes = state[fc_key].size(0)

    # decide which net fits the keys ------------------------------
    if any(k.startswith("features.0.") for k in state):          # WordCNN
        net = WordCNN(num_classes)
    elif any(k.startswith("backbone.0.") for k in state):        # WordResNet
        net = WordResNet(num_classes)
    else:                                                        # plain ResNet-18
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)


    net.load_state_dict(state, strict=True)
    net.to(device).eval()

    if idx2label is None:
        idx2label = {i: f"class-{i}" for i in range(num_classes)}
    return net, idx2label



@torch.no_grad()
def predict_topk(img_or_path, model, idx2label,
                 device: str, k: int = 10):
    """Return a (kx2) list: [(word, prob), …] sorted by prob desc."""
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path                       # already a PIL image

    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    logits = model(tensor).squeeze(0)
    probs  = logits.softmax(0)

    top = torch.topk(probs, k=min(k, probs.numel()))
    return [(idx2label[idx.item()], top.values[i].item())
            for i, idx in enumerate(top.indices)]
    
def predict(img_or_path, model, idx2label, device: str):
    return predict_topk(img_or_path, model, idx2label, device, k=1)[0][0]


def normalise_crop(c):
    # 1) convert to single‐channel
    gray = c.convert("L")
    # 2) Otsu‐style (midpoint) threshold to black & white
    bw = gray.point(lambda p: 255 if p > 128 else 0, mode="1")
    # 3) invert so ink is black (0 → 255 background, 0 foreground)
    bw = ImageOps.invert(bw.convert("L"))
    # 4) back to RGB so your 3‐channel ResNet still works
    return bw.convert("RGB")

# -----------------------------------------------------------------------------
# 4. Build the GUI
# -----------------------------------------------------------------------------
class HandwritingApp(tk.Tk):
    def __init__(self, model, idx2label, device):
        super().__init__()
        self.title("Handwritten Word Recogniser")
        self.model, self.idx2label, self.device = model, idx2label, device

        # Widgets
        self.img_label  = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)

        self.pred_var   = tk.StringVar(value="Pick an image…")
        tk.Label(self, textvariable=self.pred_var,
                 font=("Helvetica", 16, "bold")).pack(pady=(0, 15))

        tk.Button(self, text="Open Image…", command=self.open_image)\
          .pack(pady=5)

    # ---------------------------------------------------------------------
    def open_image(self):
        file_types = [("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*")]
        fpath = filedialog.askopenfilename(title="Choose handwritten word",
                                        filetypes=file_types)
        if not fpath:
            return
        
        fpath = Path(fpath)
        
        try:
            orig = Image.open(fpath).convert("RGB")
            word_img = normalise_crop(orig) 
            
            top10 = predict_topk(word_img, self.model,
                                self.idx2label, self.device, k=10)

            # 1st entry is the predicted word
            main_pred, main_prob = top10[0]
            self.pred_var.set(f"Predicted:  {main_pred}  ({main_prob:.1%})")

            # --- print others to the console -------------
            print("\nTop-10 logits for", fpath.name)
            for w, p in top10:
                print(f"{p:6.2%}  {w}")
            print("------------------------------------")            
            

            # show the original image (scaled nicely)
            w, h = orig.size
            if w > 400:
                h = int(h * 400 / w); w = 400
                orig = orig.resize((w, h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(orig)
            self.img_label.configure(image=tk_img)
            self.img_label.image = tk_img
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

def num_out_features(m: torch.nn.Module) -> int:
    """Return the number of logits this network produces."""
    if hasattr(m, "fc"):            # plain ResNet-18
        return m.fc.out_features
    if hasattr(m, "head"):          # WordResNet
        return m.head.out_features
    if hasattr(m, "classifier"):    # WordCNN
        return m.classifier.out_features
    raise AttributeError("Can't find final layer")

# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"[error] model not found: {args.model}")

    # (a) external label map if provided
    if args.label_map is not None:
        if not args.label_map.exists():
            sys.exit(f"[error] label map not found: {args.label_map}")
        idx2label = load_label_map(args.label_map)
        model, _ = load_model(args.model, device=args.device)  # ignore internal map
        if len(idx2label) != num_out_features(model):
            print("[warning] label map size "
                  f"({len(idx2label)}) ≠ model classes ({model.fc.out_features})")
    # (b) else rely on checkpoint contents / fallback
    else:
        model, idx2label = load_model(args.model, device=args.device)

    app = HandwritingApp(model, idx2label, args.device)
    app.mainloop()


if __name__ == "__main__":
    main()