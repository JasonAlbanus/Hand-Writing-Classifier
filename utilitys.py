import difflib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def plot_accuracy(train_accuracy, test_accuracy,epochs_arr):
    # Plot accuracy
    plt.subplot(1, 1, 1)
    plt.plot(epochs_arr, train_accuracy, label=f'Training Accuracy', color='blue')
    plt.plot(epochs_arr, test_accuracy, label=f'Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_loss(train_loss, test_loss,epochs_arr):
    #Plot Loss
    plt.subplot(1, 1, 1)
    plt.plot(epochs_arr, train_loss, label=f'Training Loss', color='blue')
    plt.plot(epochs_arr, test_loss, label=f'Testing Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_rate(learning_rate, epochs_arr):
    plt.subplot(1, 1, 1)
    plt.plot(epochs_arr, learning_rate, label=f'Learning Rate', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_error_chars(ground_truths,predictions):
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()

    for gt, pred in zip(ground_truths, predictions):
        matcher = difflib.SequenceMatcher(None, gt, pred)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                for c1, c2 in zip(gt[i1:i2], pred[j1:j2]):
                    substitutions[(c1, c2)] += 1
            elif tag == "delete":
                for c in gt[i1:i2]:
                    deletions[c] += 1
            elif tag == "insert":
                for c in pred[j1:j2]:
                    insertions[c] += 1
     # Plot top deletions
    top_deletions = deletions.most_common(5)
    chars, counts = zip(*top_deletions)
    plt.bar(chars, counts)
    plt.title("Top 10 Most Deleted Characters")
    plt.show()


def make_heatmap(ground_truths,predictions):
    # Track substitutions
    subs = Counter()

    # Go through all string pairs
    for gt, pred in zip(ground_truths, predictions):
        matcher = difflib.SequenceMatcher(None, gt, pred)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                for c1, c2 in zip(gt[i1:i2], pred[j1:j2]):
                    subs[(c1, c2)] += 1

    # Get sorted list of unique characters from all substitutions
    actual_chars = sorted(set(k[0] for k in subs))
    predicted_chars = sorted(set(k[1] for k in subs))

    # Create label index maps
    actual_idx = {c: i for i, c in enumerate(actual_chars)}
    pred_idx = {c: i for i, c in enumerate(predicted_chars)}

    # Initialize matrix
    matrix = np.zeros((len(actual_chars), len(predicted_chars)), dtype=int)

    # Fill matrix
    for (a, p), count in subs.items():
        i = actual_idx[a]
        j = pred_idx[p]
        matrix[i][j] = count

    # Plot it
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d",
                xticklabels=predicted_chars,
                yticklabels=actual_chars,
                cmap="Blues")
    plt.xlabel("Predicted Character")
    plt.ylabel("Actual (Ground Truth) Character")
    plt.title("Character-Level Substitution Confusion Matrix")
    plt.tight_layout()
    plt.show()


#testing
ground_truths = [
    "fight", "the", "quick", "green", "bike", "echo", "dreamer", "loud", "beach", "brown",
    "fog", "sand", "wild", "air", "cloudy", "greenish", "silence", "dark", "happy", "purple",
    "shadow", "cake", "mountain", "laughing", "lazy", "dance", "tiger", "tree", "sing",
    "study", "make", "smile", "hope", "field", "soft", "love", "fast", "lightning", "fish",
    "sun", "silver", "quickly", "journey", "peacefully", "apple", "car", "yellow", "build",
    "beautiful", "go", "quietly", "fear", "sleep", "river", "slow", "old", "yell", "snow",
    "gold", "run", "rock", "young", "faster", "wind", "star", "mountain", "peaceful", "music",
    "strong", "songbird", "bright", "jump", "red", "tired", "song", "planet", "coffee", "stone",
    "sea", "hard", "moon", "start", "rain", "space", "create", "hill", "jump", "water", "road",
    "wave", "white", "forest", "fight", "black", "brave", "fly", "green", "cat", "light", "soar"
]

predictions = [
    "fight", "ghe", "quick", "gryen", "blke", "echo", "dreamer", "lood", "beach", "brown",
    "fog", "sfnd", "wild", "air", "cloudy", "greenush", "silence", "dork", "happy", "purple",
    "shasow", "cage", "mountain", "laumhing", "lazy", "dance", "tiger", "trae", "sing",
    "study", "make", "smibe", "hope", "field", "slft", "love", "fast", "ligptning", "fish",
    "sun", "silver", "quickly", "journey", "peacefully", "apple", "car", "yellow", "build",
    "beautidul", "go", "quietly", "ffar", "sleet", "river", "slow", "old", "yell", "snow",
    "gotd", "run", "rock", "young", "faster", "wind", "star", "mountain", "peaceful", "music",
    "styong", "songbird", "briwht", "jump", "rqd", "tired", "song", "planet", "coffee", "stone",
    "sea", "hard", "moon", "start", "rain", "space", "crvate", "hill", "jump", "water", "road",
    "wave", "white", "forest", "fight", "black", "brave", "fly", "green", "cat", "light", "soar"
]
make_heatmap(ground_truths, predictions)
get_error_chars(ground_truths, predictions)
