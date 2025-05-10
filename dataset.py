import os
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True    

def right_pad_collate(batch):
    """
    Pads images in a batch to match the width of the widest image by adding 
    padding of white pixels on the right.
    
    Args:
        batch (list): List of tuples containing (image, label) pairs. 
                      Images are expected to be torch tensors with shape (channels, height, width).
    Returns:
        tuple: Contains:
            - torch.Tensor: Batch of padded images with shape (batch_size, channels, height, max_width)
            - torch.Tensor: Tensor of corresponding labels
    """
    
    # unzip the batch into separate lists of images and labels
    imgs, labels = zip(*batch)
    
    # get width of each image
    widths = [img.shape[2] for img in imgs]
    max_w = max(widths)
    
    # pad each image to match the widest image in the batch
    padded = []
    for img, w in zip(imgs, widths):
        padding = (0, max_w - w, 0, 0)  # pad right side to match max width
        padded_img = torch.nn.functional.pad(img, padding, value=1.0)
        padded.append(padded_img)
    
    # stack images into a batch tensor and convert labels to tensor
    return torch.stack(padded), torch.tensor(labels)


class HandwritingDataset(Dataset):
    """A PyTorch Dataset for loading handwritten word images and their labels.
    
    The dataset reads from a directory structure containing PNG images of handwritten words
    and their corresponding labels from an ASCII file. It filters out invalid images and
    non-alphabetic labels.
    
    Args:
        root_dir (str): Root directory containing the dataset structure
        transform (callable, optional): Optional transform to be applied to the images
    """
    def __init__(self, root_dir, transform=None):
        print("[dataset] importing the dataset")
        self.root_dir  = root_dir
        self.transform = transform
        ascii_path     = os.path.join(root_dir, "ascii", "words.txt")

        # unfiltered tuples for (rel_path_to_image, label)
        raw_samples = []          
        
        # due to poor formating in the dataset mapping file, several tweaks 
        # needed for the parsing process
        with open(ascii_path, "r", encoding="utf-8") as f:
            for line in f:
                # Ignore lines that start with a comment 
                if line.startswith("#"):        
                    continue
                
                cols = line.strip().split()
                
                # ignore rows that contain 'err' (dataset irregularities)
                if not cols or cols[1] != "ok": 
                    continue

                # parse image ID ("a01-117-05-02") and its label
                img_id = cols[0]                
                label  = cols[-1]

                # create the path components to build the relative path to the 
                # PNG.

                # as an example:
                #       img_id = "a01-117-05-02":
                #           subdir1 = "a01"
                #           subdir2 = "a01-117"
                # 
                #       final path: root_dir/words/subdir1/subdir2/img_id.png
                #                   root_dir/words/a01/a01-117/a01-117-05-02.png
                
                subdir1   = img_id.split("-")[0]   
                subdir2   = "-".join(img_id.split("-")[:2])    
                path   = os.path.join(
                    root_dir, "words", subdir1, subdir2, f"{img_id}.png"
                )
                raw_samples.append((path, label))

        # pattern to match to filter out punctuation marks from dataset
        pattern = re.compile(r'^[A-Za-z]+$')
        
        # aggregates the samples with valid images and w/o punctuation symbols 
        good_samples = []
        
        for path, label in raw_samples:
            try:
                # filter out unreadable PNGs (dataset reliability issues)
                with Image.open(path) as im:
                    im.verify()
                
                # ignore punctuation labels (messes up accuracy results)
                if pattern.match(label):
                    good_samples.append((path, label))
                    
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue

        # generate a set of unique labels to build the mapping
        unique_labels = set()
        for _, label in good_samples:
            unique_labels.add(label)
            
        # sort them alphabetically
        labels = sorted(unique_labels)
        
        # create mapping from label to index
        self.label2idx = {}
        for index, label in enumerate(labels):
            self.label2idx[label] = index
            
        # convert samples to use indices instead of text labels
        self.samples = []
        for path, label in good_samples:
            index = self.label2idx[label]
            self.samples.append((path, index))

        dropped = len(raw_samples) - len(self.samples)
        if dropped:
            print(f"[dataset] skipped {dropped} images")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        # get image path and label index from samples list
        path, label = self.samples[idx]
        
        # attempt to load and convert image to RGB format
        try:
            img = Image.open(path).convert("RGB")
            
        except (UnidentifiedImageError, OSError):
            # if image is corrupted/unreadable, recurse for a random sample 
            # instead
            return self.__getitem__(random.randrange(len(self)))
            
        # apply transforms if created
        if self.transform:
            img = self.transform(img)
            
        # return the image-label pair for training
        return img, label



def get_dataloaders(train_split: float=0.8, is_benchmark: bool=True):
    """
    Creates and returns training and testing DataLoader objects for the 
    handwriting dataset.

    Args:
        train_split (float): fraction of data for training 
        is_benchmark (bool): uses fixed seed if true for reproducible splits 

    Returns:
        tuple: (train_loader, test_loader)
    """
    print(f"[dataset] using a training split of {(train_split * 100) // 1}")

    # initialize the dataset without any transforms (to be initialized 
    # externally)
    dataset = HandwritingDataset(
        root_dir='./handwriting-dataset',
        transform=None
    )
    print("[dataset] finished importing dataset")

    # calculate sizes for train/test split
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    # set random generator: fixed seed for benchmarks, random for training
    if is_benchmark:
        rand_generator = torch.Generator().manual_seed(1)
    else: 
        rand_generator = torch.Generator()
        
    # split dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, 
                                                [train_size, test_size], 
                                                generator=rand_generator)

    # propagate label mapping to split datasets
    for sub in (train_dataset, test_dataset):
        sub.label2idx = dataset.label2idx
        
    # configure GPU memory pinning if available to speed up training
    pin_mem = torch.cuda.is_available()

    # Create training data loader with shuffling enabled
    train_loader = DataLoader(train_dataset,
                        batch_size=16,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=right_pad_collate,
                        pin_memory=pin_mem)

    # create testing data loader without shuffling
    test_loader = DataLoader(test_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=right_pad_collate,
                            pin_memory=pin_mem)

    print("[dataset] done!")
    return train_loader, test_loader
