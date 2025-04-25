import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import is_tensor


"""
This module is a dataset class specific for handwriting recognition. 

It uses PyTorch's Dataset class to load images and their corresponding labels 
from a CSV file. Calls to __getitem__ will return a tuple of images and their 
labels. 

We use the DataLoader to handle most of the heavy lifting, including shuffling 
and batching.
"""
class HandwritingDataset(Dataset):
    """Initializes the dataset for handwriting recognition

    Args:
        csv_file (str): Path to the csv file with labels
        img_dir (str): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample. 
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    """Returns the number of samples in the dataset
    
    Returns:
        int: Number of samples in the dataset
    """
    def __len__(self):
        return len(self.labels)

    """Returns a tuple of (image, label) for the given index
    
    Args:
        idx (int): Index of the sample to be fetched
        
    Returns:
        tuple: (image, label)
    """
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.labels.iloc[idx, 0]
        img_label = self.labels.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)
        
        img = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            img = self.transform(img)

        return (img, img_label)
    
    
"""Returns a DataLoader for the handwriting dataset. It uses the ResNet 
   transform specifications for image preprocessing. This goes in line with what 
   we had in the project proposal.

   Returns:
         DataLoader: A PyTorch DataLoader object for the dataset
"""
def get_dataloader():
    # Define the transformations for the dataset using the ResNet specs  
    # The mean and std values are based on the ImageNet dataset, which i would 
    # assume is fine for this as well. It may need to be tweaked
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # This matches ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset root directory
    root = './handwriting-dataset'

    # Create the dataset
    dataset = HandwritingDataset(
        csv_file=f"{root}/english.csv", 
        root_dir=f"{root}",
        transform=transform
    )

    # Create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4 
        # pin_memory=True (if we wanna throw it on a GPU)
    )

    return dataloader

