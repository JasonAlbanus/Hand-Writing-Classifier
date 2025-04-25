from dataset import get_dataloader
import matplotlib.pyplot as plt
import numpy as np

def show_dataloader_samples(num_samples=5):
    # Load the dataset 
    dataloader = get_dataloader()
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Display the specified number of samples
    for i in range(min(num_samples, len(images))):
        img = images[i].numpy().transpose((1, 2, 0))  # Convert to HWC format
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)  # Unnormalize

        plt.figure()
        plt.imshow(img)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
        
    plt.show()

if __name__ == "__main__":
    show_dataloader_samples()