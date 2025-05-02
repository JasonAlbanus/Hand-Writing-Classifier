import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HandwritingDataset(Dataset):
    """
    IAM word-level dataset loader.
    
    Reads ./handwriting-dataset/ascii/words.txt to get (image_path, label), 
    builds nested file paths under ./words, and then maps each unique 
    label to an integer label.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # read the words file containing the correct labels to use 
        ascii_path = os.path.join(root_dir, 'ascii', 'words.txt')
        
        # using list to hold (image_path, label) 
        samples = []
        with open(ascii_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_id = parts[0]          
                label = parts[-1]   
                # collect the parts to build the image path 
                parts  = img_id.split('-')          
                subdir1 = parts[0]            
                subdir2 = "-".join(parts[:2])     

                # creates the directory for the image as specified 
                #   -> ./handwriting-dataset/words/XYZ/ABC-DEF/<image>.png
                img_path = os.path.join(
                    root_dir,
                    'words',
                    subdir1,
                    subdir2,
                    img_id + '.png'
                )
                
                # add the image to the samples 
                samples.append((img_path, label))

        # create a sorted list of unique labels
        all_labels = []
        for _, label in samples:
            if label not in all_labels:
                all_labels.append(label)
                
        all_labels.sort()

        # creates a dictionary to store label to index mapping. this makes 
        # retrieval constant time for any sample
        self.label2idx = {}
        for idx in range(len(all_labels)):
            self.label2idx[all_labels[idx]] = idx

        # convert text labels to integer labels. 
        self.samples = []
        for path, label in samples:
            int_label = self.label2idx[label]
            self.samples.append((path, int_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get the image path and label for the given index from our samples list
        img_path, label = self.samples[idx]
        
        # open the image file and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        
        # apply any transformations to the image if specified 
        if self.transform:
            img = self.transform(img)
            
        return (img, label)


def get_dataloader():
    # ResNet‐style transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    dataset = HandwritingDataset(
        root_dir='./handwriting-dataset',
        transform=transform
    )

    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
