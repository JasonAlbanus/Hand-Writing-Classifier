from dataset import get_dataloader
import matplotlib.pyplot as plt
import numpy as np

def show_dataloader_samples(num_samples=5):
    # load the dataloader and access the raw dataset
    dataloader = get_dataloader()
    dataset = dataloader.dataset

    # create a dictionary to map numeric label indices back to their text labels
    # dataset.label2idx contains mappings from text -> index, so we reverse it
    idx2label = {}
    for text_label, index in dataset.label2idx.items():
        idx2label[index] = text_label

    # get random indices from the dataset
    all_indices = np.arange(len(dataset))
    chosen = np.random.choice(all_indices, size=min(num_samples, len(dataset)), replace=False)

    # for each random index, pull (img_tensor, int_label) from the dataset
    for idx in chosen:
        img_tensor, int_label = dataset[idx]

        # un-normalize the data and convert to HWC numpy array for plotting
        img = img_tensor.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)

        # display the data
        plt.figure()
        plt.imshow(img)
        plt.title(f"Transcription: “{idx2label[int_label]}”\n(Label idx: {int_label})")
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    show_dataloader_samples(5)
