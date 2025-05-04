"""
predict-handwriting.py
----------------------

Small Tkinter GUI for one-image handwriting recognition (IAM word level).

- Loads a fine-tuned model once at startup (choose cnn or cnn2).
- Lets the user pick an image via a file-chooser.
- Shows the image and predicted word in the window.

Usage
-----
$ python predict-handwriting.py cnn2
"""
import argparse
import json
from pathlib import Path
import sys
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------------------------------------------------------------
# Preprocessing pipeline (match training)
# -----------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Lambda(lambda img:
        transforms.functional.resize(
            img, size=(128, int(img.width * 128 / img.height)))
    ),
    transforms.Pad((0, 0, 16, 0), fill=255),
    transforms.ToTensor(),
])

# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
class WordResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.conv1.stride = (1, 1)
        net.layer4[0].conv1.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)

class WordCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

# -----------------------------------------------------------------------------
# Load label map
# -----------------------------------------------------------------------------
def load_label_map(path: Path):
    with path.open() as f:
        label2idx = json.load(f)
    return {int(v): k for k, v in label2idx.items()}

# -----------------------------------------------------------------------------
# Load model (cnn or cnn2)
# -----------------------------------------------------------------------------
def load_model(model_type: str, device: str):
    base = Path(__file__).parent
    weights = base / 'pre-trained' / model_type / 'model.dump'
    if not weights.exists():
        sys.exit(f"[error] model not found: {weights}")

    # load raw state_dict (may be AveragedModel state)
    state = torch.load(weights, map_location='cpu')
    # if nested checkpoint, unpack
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # strip unwanted keys and prefixes
    cleaned = {}
    for k, v in state.items():
        if k == 'n_averaged':
            continue
        # drop DataParallel prefix
        key = k.replace('module.', '')
        cleaned[key] = v
    state = cleaned

    # load label map
    label_map = load_label_map(base / 'label_map.json')

    # infer classes from head weight
    fc_weight = next(v for k,v in state.items() if k.endswith('weight') and v.ndim==2)
    num_classes = fc_weight.size(0)

    # instantiate correct net
    if model_type == 'cnn2':
        net = WordResNet(num_classes)
    else:
        net = WordCNN(num_classes)

    # load weights, move to device
    net.load_state_dict(state, strict=True)
    net.to(device).eval()

    return net, label_map

# -----------------------------------------------------------------------------
# Prediction (filtered to alphabetic words only)
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_topk(img: Image.Image, model, idx2label, device: str, k: int = 10):
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    logits = model(tensor).squeeze(0)
    probs = logits.softmax(0)
    top = torch.topk(probs, k=min(k, probs.numel()))
    # build candidates and filter
    candidates = [
        (idx2label[i.item()], top.values[j].item())
        for j, i in enumerate(top.indices)
    ]
    # only keep purely alphabetic labels
    filtered = [(w, p) for w, p in candidates if re.fullmatch(r'[A-Za-z]+', w)]
    return filtered[:k]


# -----------------------------------------------------------------------------
# Image normalization (binarize & invert)
# -----------------------------------------------------------------------------
def normalise_crop(c: Image.Image) -> Image.Image:
    gray = c.convert('L')
    bw = gray.point(lambda p: 255 if p > 128 else 0, mode='1')
    bw = ImageOps.invert(bw.convert('L'))
    return bw.convert('RGB')

# -----------------------------------------------------------------------------
# GUI Application
# -----------------------------------------------------------------------------
class HandwritingApp(tk.Tk):
    def __init__(self, model, idx2label, device):
        super().__init__()
        self.title('Handwritten Word Recogniser')
        self.model, self.idx2label, self.device = model, idx2label, device
        self.img_label = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)
        self.pred_var = tk.StringVar(value='Pick an image…')
        tk.Label(self, textvariable=self.pred_var,
                 font=('Helvetica',16,'bold')).pack(pady=(0,15))
        tk.Button(self, text='Open Image…', command=self.open_image).pack(pady=5)

    def open_image(self):
        fpath = filedialog.askopenfilename(
            title='Choose handwritten word',
            filetypes=[('Images','*.png *.jpg *.jpeg *.bmp'),('All','*')]
        )
        if not fpath:
            return
        orig = Image.open(fpath).convert('RGB')
        word_img = normalise_crop(orig)
        top10 = predict_topk(word_img, self.model,
                             self.idx2label, self.device, k=10)
        main, prob = top10[0]
        self.pred_var.set(f"Predicted:  {main}  ({prob:.1%})")
        print(f"\nTop-10 logits for {Path(fpath).name}")
        for w,p in top10:
            print(f"{p:6.2%}  {w}")
        print('-'*36)
        w,h = orig.size
        if w>400:
            h = int(h*400/w); w = 400
            orig = orig.resize((w,h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(orig)
        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['cnn','cnn2'],
                   help="Which pretrained model to use (cnn or cnn2)")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    model, idx2label = load_model(args.model, args.device)
    app = HandwritingApp(model, idx2label, args.device)
    app.mainloop()

if __name__ == '__main__':
    main()