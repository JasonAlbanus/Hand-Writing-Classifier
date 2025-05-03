from dataset import get_dataloaders
import matplotlib.pyplot as plt
import torch
import numpy as np

def run_samples(dataset, num_samples):
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    idx2label = {idx: lab for lab, idx in base_ds.label2idx.items()}
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    fig, axs = plt.subplots(1, len(indices), figsize=(4*len(indices), 4))
    axs = [axs] if len(indices) == 1 else axs

    for idx, ax in zip(indices, axs):
        img_tensor, label = dataset[idx]
        img = img_tensor.numpy().transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + 
                     np.array([0.485, 0.456, 0.406]), 0, 1)
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"'{idx2label[label]}'\n(idx: {label})", pad=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_dataloader_samples(num_samples=5):
    train_dataloader, test_dataloader = get_dataloaders()
    run_samples(train_dataloader.dataset, num_samples)
    run_samples(test_dataloader.dataset, num_samples)

    
if __name__ == "__main__":
    show_dataloader_samples(5)
