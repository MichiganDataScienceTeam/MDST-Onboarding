"""
MDST Real vs Fake Face Detection Project - Winter 2024

dataset.py - Starter code for loading and preprocessing the dataset.
"""

from pathlib import Path
from typing import Union, Callable
import pandas as pd
import torch
import numpy.typing as npt
import torchvision
from torchvision import transforms
import imageio.v3 as iio


def preprocess(image: npt.ArrayLike) -> torch.Tensor:
    """
    Preprocesses an image by applying a series of transformation.

    Args:
        image (npt.ArrayLike): The input image to be preprocessed.

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    # Convert image to tensor
    tensor = torch.tensor(image, dtype=torch.float32)

    # TODO: Edit this function to more preprocessing steps to improve model performance.
    return tensor


class RvFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        data_directory: Union[str, Path] = "data/rvf10k",
        preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
    ):
        self.data_directory = Path(data_directory)
        self.metadata = pd.read_csv(self.data_directory / f"{split}.csv")

        if preprocessor is None:
            self.preprocessor = torchvision.transforms.ToTensor()
        else:
            self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        image_metadata = self.metadata.iloc[index]
        path = self.data_directory / image_metadata["path"]
        image = self.preprocessor(iio.imread(path))
        return image, image_metadata["label"]


def get_loaders(
    batch_size: int = 32,
    preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = preprocess,
    pin_memory: bool = False,
    data_directory: Union[str, Path] = "data/rvf10k",
    num_workers: int = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns the DataLoader objects for the training and validation sets.

    Args:
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of workers for the DataLoader. Defaults to None.
        pin_memory (bool): Whether to pin memory in the DataLoader. Defaults to False.
        data_directory (Union[str, Path]): The directory where the data is stored.
        preprocessor (Callable[[npt.ArrayLike], torch.Tensor]): The preprocessor function to use.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The DataLoader objects for the training and validation sets.
    """
    if num_workers is not None:
        train_loader = torch.utils.data.DataLoader(
            RvFDataset("train", data_directory, preprocessor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            RvFDataset("valid", data_directory, preprocessor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            RvFDataset("train", data_directory, preprocessor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )

        val_loader = torch.utils.data.DataLoader(
            RvFDataset("valid", data_directory, preprocessor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader
