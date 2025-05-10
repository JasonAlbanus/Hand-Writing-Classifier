import json
import time
from typing import Tuple, Dict, List
import dataset
import random
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# defaults from ImageNet 
IMAGENET_MEAN = [0.485,0.456,0.406]   
IMAGENET_STD = [0.229,0.224,0.225]

def pad_right_to(img, target_w=320):
    """Pad on the right so every image is target_w x 128."""
    if img.width >= target_w:
        return img
    
    pad = target_w - img.width
    return transforms.functional.pad(img, (0, 0, pad, 0), fill=255)


class WordCNN(torch.nn.Module):
    """
    CNN architecture for handwriting recognition based on ResNet18.
    
    Key changes from standard ResNet:
        - First conv layer stride reduced from 2 to 1 to preserve fine details
        - Last residual block stride reduced to prevent excessive downsampling
        - Added DropBlock regularization for structured dropout
        - Modified final classification head for handwriting classes
    
    Args:
        num_classes (int): Number of output classes
        dropout_p (float): Dropout prob for final layer
    """
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        # load pretrained ResNet18 with ImageNet weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # modify first conv layer stride from (2,2) to (1,1)
        # This preserves more spatial information from input images, critical 
        # for handwriting where fine details matter
        net.conv1.stride = (1, 1)
        
        # modify final residual block (layer4) stride from 2 to 1
        # prevents excessive downsampling of feature maps. both conv layer and 
        # downsample projection need stride adjustment
        net.layer4[0].conv1.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        
        # extract all layers except final FC layer
        self.backbone = torch.nn.Sequential(*list(net.children())[:-1])
        
        # add dropblock, more effective than standard dropout 
        self.dropblock = DropBlock2d(p=0.1, block_size=5)
        
        # final classification head using ResNet's feature dimension
        self.head = torch.nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        # extract features through ResNet backbone
        x = self.backbone(x)
        
        # apply DropBlock regularization
        x = self.dropblock(x)
        
        # flatten spatial dimensions for linear layer
        x = x.flatten(1)
        
        # final classification
        return self.head(x)

class DropBlock2d(torch.nn.Module):
    """
    DropBlock is a structured form of dropout that drops contiguous blocks of 
    feature map values. It helps regularize the CNN better than the standard.
    
    Args:
        p (float): probability of dropping a block
        block_size (int): size of blocks to drop
    """
    def __init__(self, p: float = 0.1, block_size: int = 5):
        super().__init__()
        self.p = p
        self.block_size = block_size

    def forward(self, x):
        # don't apply during evaluation or if probability is 0
        if not self.training or self.p == 0.0:
            return x
        
        # convert drop probability to gamma, accounting for block size
        gamma = self.p / (self.block_size ** 2)
        
        # create a mask with the same batch size and height/width as input x,
        # but only for one channel (we'll apply it to all channels later using 
        # max pool)
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        
        # generate random values between 0 and 1
        random_values = torch.rand(batch_size, 1, height, width, device=x.device)
        
        # create binary mask: 
        #   1 where random value < gamma (will be dropped), 
        #   0 otherwise
        mask = (random_values < gamma)
        
        # convert boolean mask to float for tensor
        mask = mask.float()
        
        # expand dropped regions to (block_size x block_size) using max pooling
        kernel_size = self.block_size
        padding = kernel_size // 2
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # apply mask by setting dropped regions to 0
        return x * (1 - mask)

def mixup(x, y, alpha: float = 0.2):
    """
    Applies the mixup augementation to the x input tensor (of batch size N) and 
    the respective N labels in the y tensor. Alpha determines the value to use 
    in the Beta distribution to yield the lambda in [0, 1]. Small alpha yields 
    lambdas closer to the 0 to 1 edges, larger alphas yield lambas closer to 0.5 
    
    Smaller lambda values will express the original inputs greater in the mixed
    output, whereas larger lambda values will express the permuted inputs 
    greater. In effect, we want the model to claim the probability of "y[i]" 
    being the label for some i-th component of the mixed_x output of being 
    lambda.   

    Args:
        x (tensor): batched input tensor (batch size of N)
        y (tensor): labels for batched input 
        alpha (float, optional): Scalar to determine lambda. Defaults to 0.2.

    Returns:
        mixed_x (tensor): batch of blended images
        (y, y[idx], lam) (tensor, tensor, float): y contains the original labels
                                                  y[idx] is the permuted labels
                                                  lam is the scalar weight 
    """
    lam = np.random.beta(alpha, alpha)
    
    # Tensor of length N containing the random permutation of [0, ..., N - 1]
    #   We can use it to "index" a tensor or permute the N batches of images 
    #   into the permutation. Doing so optimizes for the vectorized operation.   
    idx = torch.randperm(x.size(0), device=x.device)
    
    # Combine the original inputs with the permutation to generate a mixed input 
    # tensor. The original inputs have a mixing coefficient of lambda.
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, (y, y[idx], lam)

# compute one epoch
def epoch_loop(model, loader, criterion, optimizer, device, scaler,
               phase: str, log_every: int = 100) -> Tuple[float, float]:
    """
    Performs one training/validation epoch.
    
    Args:
        model: The neural network model
        loader: DataLoader containing batches of images and labels
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (AdamW) 
        device: Device to run on (CPU/GPU)
        scaler: Gradient scaler for mixed precision training
        phase: Either "train", "val" or "clean" to control model behavior
        log_every: How often to print progress
    
    Returns:
        Tuple of (average loss, accuracy) for the epoch
    """
    
    # set model to training or eval mode
    is_train = (phase == "train")
    model.train(is_train)
    
    # track running stats
    running_loss = 0 
    correct = 0 
    total = 0
    
    # iterate through batches
    for i, (x, y) in enumerate(loader, 1):
        # move data to device
        x = x.to(device) 
        y = y.to(device)
        
        # apply mixup augmentation during training with 80% probability
        if is_train and random.random() < 0.8:
            x, y = mixup(x, y, alpha=0.2)
            
        # forward pass with automatic mixed precision
        with autocast(device.type, enabled=(device.type == "cuda")):
            # get model predictions
            out = model(x)
            
            # calculate loss (handle both regular and mixup cases)
            if isinstance(y, tuple):
                # mixup case: weighted combination of two labels
                y1, y2, lam = y
                loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
            else:
                # regular case: single label
                loss = criterion(out, y)
                
        # backward pass and optimization (training only)
        if is_train:
            # clear the gradients
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                # mixed precision training steps
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # regular training steps
                loss.backward()
                optimizer.step()
                
        # update stats for data dump
        running_loss += loss.item() * x.size(0)
        y_true = y[0] if isinstance(y, tuple) else y
        
        # get predicted classes
        preds = out.argmax(1) 
         
        # count correct predictions
        correct += preds.eq(y_true).sum().item()  
        total += y_true.size(0)
        
        # print progress periodically
        if log_every and i % log_every == 0:
            print(f"{phase:5} [{i}/{len(loader)}] loss {loss.item():.3f}")
            
    # return average loss and accuracy for the epoch        
    return running_loss / total, correct / total

def train(num_epochs: int = 25, stop_threshold: int = 5, is_benchmark: bool = False):
    """
    Trains a CNN model for handwriting recognition using PyTorch.
    
    The training process includes the following optimizations:
        - Residual Network (ResNet18) CNN architecture and ImageNet weights
        - Cross Entropy loss with label smoothing
        - adam optimizer with weight decay (AdamW)
        - learning rate scheduling 
        - Stochastic Weight Averaging (SWA)
        - early stopping on stagnation
        - mixed precision training with CUDA
        
    Args:
        num_epochs (int, optional): epochs to train for
        stop_threshold (int, optional): epochs to wait for val gains before stopping
        is_benchmark (bool, optional): disables seed for benchmarking if true
        
    Outputs:
        - best model weights to './model.dump'
        - label mapping to './label_map.json'
        - training history to './history.json'
    """
    
    if not is_benchmark:
        # training with a seed (disabled for benchmarks)
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # load the dataset from local files into torch dataloaders
    train_dataloader, validation_dataloader = dataset.get_dataloaders()
    
    # unwrap subset wrappers to get the underlying HandwritingDataset
    base_train_ds = train_dataloader.dataset.dataset
    base_val_ds   = validation_dataloader.dataset.dataset

    # training transforms:
    #   I. resize image to height 128px, preserving aspect ratio
    #  II. pad right with white pixels to fixed width (320px)
    # III. random affine transforms for augmentation:
    #       > rotation of +/- 4 degrees
    #       > translation of +/- 3% in x and y
    #       > scale by 95-105% of original size
    #  IV. convert to tensor (scales to 0-1)
    #   V. normalize using ImageNet stats for transfer learning
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.resize(
                                      img, size=(128, 
                                      int(img.width * 128 / img.height)))),
        transforms.Lambda(pad_right_to),
        transforms.RandomAffine(degrees=4,
                                translate=(0.03, 0.03),
                                scale=(0.95, 1.05),
                                fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # transforms for validation and clean training visualization runs.
    # no augmentations included! this data is only used to generate loss/
    # accuracy metrics for visualization in the history JSON, not for actual 
    # training or the final model dump
    clean_run_transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.resize(
                                      img, size=(128, 
                                      int(img.width * 128 / img.height)))),
        transforms.Lambda(pad_right_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


    # apply the transforms to the datasets
    base_train_ds.transform = train_transform
    base_val_ds.transform = clean_run_transform 
    
    num_classes = len(train_dataloader.dataset.label2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # create our model and send it to the device 
    model = WordCNN(num_classes).to(device)
    
    # using CE loss with label smoothing to improve generalization
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Initialize the optimizer with AdamW, which is a variant of Adam that 
    # includes weight decay for better regularization & to prevent overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00005)
    
    
    # Stochastic Weight Averaging (SWA) helps improve generalization by 
    # averaging multiple points along the trajectory of SGD
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.0002)
    swa_start = 25  # epoch to start SWA 
    
    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
    
    
    # initialize gradient scaler for mixed precision training on GPU to speed up
    # training
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    # keep track of the best validation accuracy seen so far 
    best_val = 0.0
    best_val_epoch = 0
    
    # counter for epochs without improvement (for early stopping)
    wait = 0 
    gain_threshold = 0.02 # require 2% jumps for counter reset
    
    # using a dictionary to store history, output as json after training/
    # validation    
    history: Dict[str, List[float]] = {
        "train_loss": [], 
        "train_accuracy": [], 
        "validation_loss": [], 
        "validation_accuracy": []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # train the model with the optimizations. save the average loss 
        tl, _ = epoch_loop(model, train_dataloader, criterion, optimizer, device, scaler, "train")

        # set the transform for the clean training runs (for stats)
        base_train_ds.transform = clean_run_transform
        
        # save the accuracy for the unoptimized training run.
        _, ta = epoch_loop(model, train_dataloader, criterion,
                           optimizer, device, None, "clean")
        
        # switch back to optimized transform
        base_train_ds.transform = train_transform

        # validation accuracy
        vl, va = epoch_loop(model, validation_dataloader, criterion,
                             optimizer, device, None, "val")

        # add stats to the history table
        history["train_loss"].append(tl)
        history["train_accuracy"].append(ta)
        history["validation_loss"].append(vl)
        history["validation_accuracy"].append(va)

        print(f"summary: train {ta*100:5.2f}% | val {va*100:5.2f}%")

        if va - best_val > gain_threshold:
            # hit an epoch with improved val accuracy 
            best_val = va
            best_val_epoch = epoch
            wait = 0
            # output the intermediate model if interrupts stop execution
            torch.save(model.state_dict(), "model.dump")
        else:
            # increment wait count to track stagnation
            wait += 1
            if wait >= stop_threshold:
                print("Early stopping.")
                break

        # after swa_start epochs, use SWA as scheduler
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # normal LR scheduler 
            scheduler.step()
            
    
    # save the final label map exactly once, after training loop
    with open('label_map.json', 'w') as f:
        json.dump(base_train_ds.label2idx, f)

    # finalize SWA
    base_val_ds.transform = clean_run_transform
    torch.optim.swa_utils.update_bn(validation_dataloader, swa_model, device=device)
    torch.save(swa_model.state_dict(), "model.dump")

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy {best_val*100:.2f}% (epoch {best_val_epoch})")
    print("Model weights  -> model.dump")
    print("Training curve -> history.json")

def main():
    start = time.time()
    train(is_benchmark=True)
    print(f"total time {time.time() - start}s")


if __name__ == "__main__":
    main()