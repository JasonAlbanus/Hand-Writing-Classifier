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

# ── GUI & imaging ────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# 3. Helper functions
# -----------------------------------------------------------------------------
def load_label_map(json_path: Path):
    with json_path.open() as fp:
        label2idx = json.load(fp)
    return {int(v): k for k, v in label2idx.items()}

def load_model(weights: Path, device: str):
    """Load a ResNet18 variant; return (model, idx2label_or_None)."""
    # 1) Try TorchScript first
    try:
        model = torch.jit.load(str(weights), map_location=device)
        model.eval()
        num_classes = model(torch.empty(1, 3, 224, 224)).shape[1]
        return model, {i: str(i) for i in range(num_classes)}
    except RuntimeError:
        pass

    # 2) Vanilla state-dict
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:   # Lightning ckpt
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}

    # try to fetch label map inside checkpoint
    idx2label = extract_idx2label(state)

    # infer #classes from final layer weight
    for k, v in state.items():
        if k.endswith(".weight") and v.ndim == 2:
            num_classes = v.size(0)
            break
    else:
        raise ValueError("Could not infer number of classes from checkpoint")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    missing, _ = model.load_state_dict(state, strict=False)   # ignore extras
    if missing:
        print("[warning] missing keys:", missing[:5], "…")
    model.to(device).eval()
    if idx2label is None:
        idx2label = {i: f"class-{i}" for i in range(num_classes)}
    return model, idx2label


@torch.no_grad()
def predict(path: Path, model, idx2label, device: str) -> str:
    img = Image.open(path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    logits = model(tensor)
    return idx2label[logits.argmax(1).item()]

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
            word = predict(fpath, self.model, self.idx2label, self.device)
            self.pred_var.set(f"Predicted text:  {word}")

            # Show the image (fit within 400px width)
            img_orig = Image.open(fpath)
            w, h = img_orig.size
            if w > 400:
                h = int(h * 400 / w)
                w = 400
                img_orig = img_orig.resize((w, h), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img_orig)
            self.img_label.configure(image=tk_img)
            self.img_label.image = tk_img           # keep a reference
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

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
        if len(idx2label) != model.fc.out_features:
            print("[warning] label map size "
                  f"({len(idx2label)}) ≠ model classes ({model.fc.out_features})")
    # (b) else rely on checkpoint contents / fallback
    else:
        model, idx2label = load_model(args.model, device=args.device)

    app = HandwritingApp(model, idx2label, args.device)
    app.mainloop()


if __name__ == "__main__":
    main()