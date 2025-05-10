import json
import matplotlib.pyplot as plt
# Utility Functions for plotting loss and accuracy
# As well a way to take results in json and plot them


# To plot the accuracy for both testing and training against epochs
def plot_accuracy(train_accuracy, test_accuracy, epochs_arr):
    # Plot accuracy
    plt.subplot(1, 1, 1)
    plt.plot(epochs_arr, train_accuracy, label=f"Training Accuracy", color="blue")
    plt.plot(epochs_arr, test_accuracy, label=f"Testing Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

# To plot the loss for both testing and training against epochs
def plot_loss(train_loss, test_loss, epochs_arr):
    # Plot Loss
    plt.subplot(1, 1, 1)
    plt.plot(epochs_arr, train_loss, label=f"Training Loss", color="blue")
    plt.plot(epochs_arr, test_loss, label=f"Testing Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Takes a json with loss + accuracy for training and validation to plot
with open('./pre-trained/cnn2/history.json', 'r') as f:
     history = json.load(f)

epochs = list(range(1, len(history['tr_loss']) + 1))
plot_accuracy(history['tr_acc'], history['val_acc'], epochs)
plot_loss(history['tr_loss'], history['val_loss'], epochs)

