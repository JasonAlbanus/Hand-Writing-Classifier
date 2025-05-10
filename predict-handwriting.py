import argparse
import json
import sys
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from cnn import WordCNN as SmallCNN
from cnn2 import WordCNN as ResNetCNN, pad_right_to

# Defaults from ImageNet for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# File and directory names for models and labels
MODEL_DIR_NAME = 'pre-trained'
LABEL_MAP_FILENAME = 'label_map.json'
MODEL_WEIGHTS_FILENAME = 'model.dump'

# GUI display settings
IMAGE_DISPLAY_MAX_WIDTH = 400

# Preprocessing pipeline for inference
PREPROCESS = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.resize(
        img, size=(128, int(img.width * 128 / img.height)))
    ),
    transforms.Lambda(lambda img: pad_right_to(img, target_w=256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_label_map(path: Path) -> dict:
    """Loads a label mapping from a JSON file and inverts it.

    This function reads a JSON file containing a mapping from labels to indices,
    and inverts it to create a mapping from indices to labels.

    Args:
        path (Path): Path to the JSON file containing the label mapping.

    Returns:
        dict: A dictionary mapping integer indices to labels.
    """
    with path.open() as f:
        label2idx = json.load(f)
    # invert mapping for index -> label
    return {int(v): k for k, v in label2idx.items()}


def _clean_state_dict(state: dict) -> dict:
    """
    Removes unwanted keys and prefixes from a model state dictionary.

    This function processes a model state dictionary by removing the 
    'n_averaged' key if present and stripping 'module.' prefix from keys.

    Args:
        state (dict): The model state dictionary to clean

    Returns:
        dict: A cleaned version of the state dictionary with unwanted keys removed
             and prefixes stripped from remaining keys
    """
    cleaned = {}
    for k, v in state.items():
        if k == 'n_averaged':
            continue
        cleaned[k.replace('module.', '')] = v
    return cleaned


def load_model(model_type: str, device: str) -> tuple[torch.nn.Module, dict]:
    """
    Loads a pre-trained model ('cnn' or 'cnn2') and its label map.

    Args:
        model_type (str): 'cnn' for the small CNN, 'cnn2' for ResNet-based CNN
        device (str): 'cpu' or 'cuda'
    Returns:
        model: initialized and loaded PyTorch model (in eval mode)
        idx2label: dict mapping class indices to human-readable labels
    """
    base_path = Path(__file__).parent
    weights_path = base_path / MODEL_DIR_NAME / model_type / MODEL_WEIGHTS_FILENAME
    if not weights_path.exists():
        sys.exit(f"[error] model not found: {weights_path}")

    # load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # unpack if using AveragedModel checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
        
    state = _clean_state_dict(checkpoint)

    # load label map
    idx2label = load_label_map(base_path / LABEL_MAP_FILENAME)

    # infer number of classes from the weight matrix of the final layer
    fc_weight = next(v for v in state.values() if v.ndim == 2)
    num_classes = fc_weight.size(0)

    # select the correct network class
    model_classes = {
        'cnn': SmallCNN,
        'cnn2': ResNetCNN,
    }
    NetClass = model_classes.get(model_type)
    if NetClass is None:
        sys.exit(f"[error] Unknown model type: {model_type}")

    model = NetClass(num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, idx2label


@torch.no_grad()
def predict_topk(img: Image.Image, model: torch.nn.Module, idx2label: dict, device: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Predicts the top-k most likely labels for a single handwritten image.

    Args:
        img (Image.Image): Input PIL image containing handwritten text
        model (torch.nn.Module): Trained PyTorch model for prediction
        idx2label (dict): Dictionary mapping indices to label strings 
        device (str): Device to run inference on ('cuda' or 'cpu')
        k (int, optional): Number of top predictions to return.

    Returns:
        list[tuple[str, float]]: List of tuples containing:
            - str: Predicted label (alphabetic characters only)
            - float: Prediction probability/confidence score
            Sorted by probability in descending order, limited to k results.

    Note:
        - Prediction is done in inference mode (torch.no_grad)
        - Only returns labels containing alphabetic characters
        - Returned list may contain fewer than k predictions if insufficient 
          valid labels found
    """
    # preprocess image to tensor, add batch dimension, and move to device
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    
    # run the model to get raw logit scores for each class
    logits = model(tensor).squeeze(0)
    
    # convert logits to probabilities using softmax
    probs = logits.softmax(0)
    
    # get k highest predictions (probabilities) 
    top = torch.topk(probs, k=min(k, probs.numel()))

    # create list of (word, probability) tuples from top predictions
    predictions = []
    for idx, prob in zip(top.indices, top.values):
        word = idx2label[idx.item()]
        probability = prob.item()
        predictions.append((word, probability))
    
    # filter to keep only alphabetic words and limit to k results
    alphabetic_predictions = []
    for word, prob in predictions:
        if re.fullmatch(r'[A-Za-z]+', word):
            alphabetic_predictions.append((word, prob))
    
    
    # feturn the first k results
    return alphabetic_predictions[:k]


def normalise_crop(img: Image.Image) -> Image.Image:
    """
    Converts and normalizes an input image to a black and white RGB format.

    Args:
        img (Image.Image): Input PIL Image object.

    Returns:
        Image.Image: Normalized black and white RGB image where pixels > 128 are 
                     set to white (255) and pixels <= 128 are set to black (0).
    """
    
    gray = img.convert('L')
    bw = gray.point(lambda p: 255 if p > 128 else 0, mode='1')
    return bw.convert('RGB')


class HandwritingApp(tk.Tk):
    """
    A Tkinter-based GUI for our handwriting model.

    Allows users to load an image of handwritten text, processes it using a pre-
    trained model, and displays the top predictions for the recognized word 
    along with their probabilities.

    Attributes:
        model: The pre-trained model used for handwritten word recognition.
        idx2label (dict): A mapping from model output indices to labels.
        device (str): The device on which the inference will be done on.
        img_label (tk.Label): Tkinter widget for displaying the loaded image.
        pred_var (tk.StringVar): Tkinter string for displaying the prediction.
    """
    def __init__(self, model, idx2label, device: str):
        super().__init__()
        self.title('Handwritten Word Recogniser')
        self.model = model
        self.idx2label = idx2label
        self.device = device

        self.img_label = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)

        self.pred_var = tk.StringVar(value='Pick an image…')
        tk.Label(self, textvariable=self.pred_var,
                 font=('Helvetica', 16, 'bold')).pack(pady=(0, 15))

        tk.Button(self, text='Open Image…', command=self.open_image).pack(pady=5)


    def open_image(self):
        """
        Opens a file dialog to select an image, processes the image, and 
        displays predictions.

        This method allows the user to select an image file containing 
        handwritten text. The image is normalized and passed through the pre-
        trained model to generate predictions. The top prediction is displayed 
        in the GUI, and the top 10 predictions are printed to the console.

        If the image is too large, it is resized to fit within a predefined 
        maximum width while maintaining aspect ratio.
        """
        fpath = filedialog.askopenfilename(
            title='Choose handwritten word',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp'), ('All', '*')]
        )
        if not fpath:
            return

        original = Image.open(fpath).convert('RGB')
        word_img = normalise_crop(original)

        top_preds = predict_topk(word_img, self.model, self.idx2label, self.device, k=10)
        if top_preds:
            w, p = top_preds[0]
            self.pred_var.set(f"Predicted:  {w}  ({p:.1%})")
        else:
            self.pred_var.set("No alphabetic prediction found.")

        print(f"\nTop-10 for {Path(fpath).name}")
        for w, p in top_preds:
            print(f"{p:6.2%}  {w}")
        print('-' * 36)

        disp = original.copy()
        w_, h_ = disp.size
        if w_ > IMAGE_DISPLAY_MAX_WIDTH:
            h_new = int(h_ * IMAGE_DISPLAY_MAX_WIDTH / w_)
            disp = disp.resize((IMAGE_DISPLAY_MAX_WIDTH, h_new), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(disp)
        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img


def main():
    parser = argparse.ArgumentParser(description="Handwritten Word Recogniser GUI")
    parser.add_argument('model', choices=['cnn', 'cnn2'],
                        help="Which pretrained model to use")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device for inference")
    args = parser.parse_args()

    model, idx2label = load_model(args.model, args.device)
    app = HandwritingApp(model, idx2label, args.device)
    app.mainloop()


if __name__ == '__main__':
    main()