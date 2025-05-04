import os, random
from PIL import Image, ImageFile, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True      # let PIL load truncated files when it can

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        ascii_path     = os.path.join(root_dir, "ascii", "words.txt")

        raw_samples = []          # (full_path, label) even if the PNG is bad
        with open(ascii_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):        # comment
                    continue
                cols = line.strip().split()
                if not cols or cols[1] != "ok": # ‘err’ rows → skip
                    continue

                img_id = cols[0]                # a01-117-05-02
                label  = cols[-1]

                sub1   = img_id.split("-")[0]   # a01
                sub2   = "-".join(img_id.split("-")[:2])    # a01-117
                path   = os.path.join(
                    root_dir, "words", sub1, sub2, f"{img_id}.png"
                )
                raw_samples.append((path, label))

        # ---------- filter out unreadable PNGs -------------------------------
        good_samples = []
        for p, lab in raw_samples:
            try:
                with Image.open(p) as im:
                    im.verify()
                good_samples.append((p, lab))
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue

        # ---------- filter out punctuation labels ----------------------------
        import re
        pattern = re.compile(r'^[A-Za-z]+$')
        good_samples = [(p, lab) for p, lab in good_samples if pattern.match(lab)]

        # build label map
        labels = sorted({lab for _, lab in good_samples})
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.samples   = [(p, self.label2idx[lab]) for p, lab in good_samples]

        dropped = len(raw_samples) - len(self.samples)
        if dropped:
            print(f"[dataset] skipped {dropped} broken images")

    # -------------------------------------------------------------------------
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # A second guard in case PIL.lazy-decode still trips on a weird file
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # pick another random sample instead of crashing
            return self.__getitem__(random.randrange(len(self)))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(train_split=0.8):
    # # ResNet‐style transforms
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std =[0.229, 0.224, 0.225]
    #     )
    # ])

    dataset = HandwritingDataset(
        root_dir='./handwriting-dataset',
        transform=None
    )

    # split dataset into training and testing datasets
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    rand_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=rand_generator)

    # make the mapping available on the subsets too
    for sub in (train_dataset, test_dataset):
        sub.label2idx = dataset.label2idx
        

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
