import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fixes improper conda environment installation

# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model
from dc1.cxrnet import CXRModel

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List

aug_num = 1

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    recache = False
    if args.force_recache:
        recache = True
    print("Loading training data")
    # Load the train and test data set
    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"), augment = True, n_augments = aug_num, test = False, cache_path=Path("dc1/data/cachetrain"), force_recache=recache)
    print("Training data loaded")
    print("Loading testing data")
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"), augment = False, test = True, cache_path=Path("dc1/data/cachetest"), force_recache=recache)
    print("Testing data loaded")
    print(test_dataset.targets)
    # Neural Net (note: set # classes here)
    model = Net(n_classes=6)
    if args.use_cxr:
        print("Using Google CXR Foundation model...")
        model = CXRModel(n_classes=6)

    # Optimizer and loss
    # optimizer = optim.SGD(model.parameeters(), lr=0.001, momentum=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    optimizer = torch.optim.Adam([
        {'params': model.base_model.parameters(), 'lr': 0.001},  # Slower learning for pretrained layers
        {'params': model.fc.parameters(), 'lr': 0.003, 'weight_decay' : 1e-4}  # Faster learning for new classifier
    ])

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # Debug mode toggles device usage
    DEBUG = False

    # Decide the device (CPU, CUDA, or MPS)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        summary(model, (1, 128, 128), device=device)
    elif torch.backends.mps.is_available() and not DEBUG:
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches, largest_aug_factor = aug_num, use_augmentation = False
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches, use_augmentation = False
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    # Train/test loop
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # mean_loss_test = float('inf')
    for e in range(n_epochs):
        if activeloop:
            # Train model
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e+1} training done, loss on train set: {mean_loss:.4f}\n")

            # Test model (prints accuracy inside test_model!)
            losses_test = test_model(model, test_sampler, loss_function, device)
            mean_loss_test = sum(losses_test) / len(losses_test)
            mean_losses_test.append(mean_loss_test)
            print(f"Epoch {e+1} testing done, loss on test set: {mean_loss_test:.4f}\n")

            # Optional: Live plotting with plotext
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")
            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])
            plotext.show()
            # if epoch > 1:
            scheduler.step(mean_loss_test)

    # Save model at the end
    now = datetime.now()
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create a matplotlib figure of the losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="number of training iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument("--balanced_batches", help="whether to balance batches for class labels",
                        default=True, type=bool)
    parser.add_argument("--use_cxr", action="store_true", help="Use Google CXR model instead of custom CNN")
    parser.add_argument("--force_recache", action="store_true",
                        help="Force recaching of processed data even if a cache already exists")
    args = parser.parse_args()
    main(args)