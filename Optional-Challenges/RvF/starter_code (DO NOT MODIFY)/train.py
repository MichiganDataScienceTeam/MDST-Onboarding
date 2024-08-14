"""
MDST Real vs Fake Face Detection Project - Winter 2024

train.py - Utilities for training and evaluating a model.
"""

from pathlib import Path
from typing import Callable, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
seaborn.set_theme()


def save_model(
    epoch: int,
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
):
    """
    Save the model to a file.

    Args:
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to be saved.
        checkpoint_dir (Path): The directory to save the model to.
    """
    checkpoint_dir = Path(checkpoint_dir) / model.__class__.__name__
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        checkpoint_dir / f"model_{epoch}.pt",
    )


def load_model(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    epoch: int,
    map_location: str = "cpu",
):
    """
    Load the model from a file.

    Args:
        model (torch.nn.Module): The model to be loaded.
        checkpoint_dir (Union[str, Path]): The directory to load the model from.
        epoch (int): The epoch number of the model to load.
        map_location (str, optional): The device to load the model on. Defaults to "cpu".

    Raises:
        FileNotFoundError: If the specified model file does not exist.

    Returns:
        None
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_file = checkpoint_dir / model.__class__.__name__ / f"model_{epoch}.pt"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model at epoch {epoch} in directory {str(model_file.parent())} does not exist."
        )

    model.load_state_dict(
        torch.load(model_file, map_location=torch.device(map_location))
    )


def evaluate(
    model: torch.nn.Module, criterion: Callable, loader: DataLoader
) -> tuple[float]:
    """
    Evaluate the performance of a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (Callable): The loss function used for evaluation.
        loader (DataLoader): The data loader containing the evaluation data.

    Returns:
        tuple[float]: A tuple containing the accuracy and average loss.

    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        loss = 0.0
        for X, y in loader:
            outputs = model(X.to(device)).to("cpu")

            loss += criterion(outputs, y).detach().sum().item()
            _, predicted = torch.max(outputs.data, 1)

            total += len(y)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total, loss / total


def train_model(
    model: torch.nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    save_checkpoint: bool = True,
    checkpoint_dir: Union[Path, str] = "checkpoints",
) -> dict[str, list[float]]:
    """
    Train a given model using the specified criterion, optimizer, and data loaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (Callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        epochs (int, optional): The number of training epochs. Defaults to 10.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and validation losses and accuracies.
    """
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    model = model.to(device)

    for epoch in range(epochs):
        train_correct, num_train_examples = 0, 0
        total_train_loss = 0.0
        model.train()

        pbar = tqdm(train_loader)
        for X, y in pbar:
            optimizer.zero_grad()
            outputs = model(X.to(device))
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()

            predicted_labels = outputs.argmax(axis=1)
            batch_correct = (predicted_labels == y.to(device)).to("cpu").sum().item()

            num_train_examples += X.shape[0]
            train_correct += batch_correct
            total_train_loss += loss.to("cpu").item()

            pbar.update(1)
            pbar.set_description(
                f"Batch Accuracy: {batch_correct / X.shape[0]:.2f} , Total Accuracy: {train_correct / num_train_examples:.2f}"
            )

        if save_checkpoint:
            save_model(epoch, model, Path(checkpoint_dir))

        train_accuracy = train_correct / num_train_examples

        train_losses.append(total_train_loss / num_train_examples)
        train_accuracies.append(train_accuracy)

        val_accuracy, val_loss = evaluate(model, criterion, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}: Loss - (Train {total_train_loss:.2f}/Val {val_loss:.2f}), "
            f"Accuracy - (Train {train_accuracy:.2f}/Val {val_accuracy:.2f})"
        )

    return {
        "loss": {
            "train": train_losses,
            "val": val_losses,
        },
        "accuracy": {
            "train": train_accuracies,
            "val": val_accuracies,
        },
    }


def plot_performance(history: dict[str, dict[str, list[float]]]) -> mpl.figure.Figure:
    """
    Plots the performance of a model during training and validation.

    Args:
        history (dict[str, dict[str, list[float]]]): A dictionary containing the performance history of the model.
            The keys of the dictionary represent the metrics, and the values are dictionaries containing the
            training and validation scores for each metric.

    Returns:
        mpl.figure.Figure: The matplotlib figure object containing the performance plot.
    """
    fig, axes = plt.subplots(len(history), 1, figsize=(15, 5))
    for i, (metric, values) in enumerate(history.items()):
        train, val = values["train"], values["val"]
        axes[i].plot(train, label="train")
        axes[i].plot(val, label="val")
        axes[i].set_title(f"{metric}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].legend()
    return fig
