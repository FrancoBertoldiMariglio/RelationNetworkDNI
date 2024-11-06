import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
from typing import List, Tuple, Optional
from torchvision import transforms


class BinaryImageDataset(Dataset):
    """Improved dataset class with better type hints and organization"""

    def __init__(self, valid_dir: str, invalid_dir: str, transform: Optional[transforms.Compose] = None):
        self.transform = transform

        # Load and validate directories
        self.valid_images = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.invalid_images = [os.path.join(invalid_dir, f) for f in os.listdir(invalid_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.valid_images or not self.invalid_images:
            raise ValueError("No valid images found in one or both directories")

        self.all_images = self.valid_images + self.invalid_images
        self.labels = torch.tensor([1] * len(self.valid_images) + [0] * len(self.invalid_images))

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.all_images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class EpisodeSampler:
    """Improved episode sampler with better sampling strategy"""

    def __init__(self, dataset: BinaryImageDataset, n_shot: int, n_query: int):
        self.n_shot = n_shot
        self.n_query = n_query

        # Pre-compute indices for each class
        self.valid_indices = torch.where(dataset.labels == 1)[0].tolist()
        self.invalid_indices = torch.where(dataset.labels == 0)[0].tolist()

        # Validate parameters
        if n_shot > min(len(self.valid_indices), len(self.invalid_indices)):
            raise ValueError("n_shot is larger than available samples in one or both classes")

        if n_query > min(len(self.valid_indices) - n_shot, len(self.invalid_indices) - n_shot):
            raise ValueError("n_query is too large given n_shot and available samples")

    def sample_episode(self) -> Tuple[List[int], List[int]]:
        # Sample support sets
        support_valid = random.sample(self.valid_indices, self.n_shot)
        support_invalid = random.sample(self.invalid_indices, self.n_shot)

        # Sample query sets from remaining samples
        remaining_valid = list(set(self.valid_indices) - set(support_valid))
        remaining_invalid = list(set(self.invalid_indices) - set(support_invalid))

        query_valid = random.sample(remaining_valid, self.n_query)
        query_invalid = random.sample(remaining_invalid, self.n_query)

        # Combine and shuffle indices
        support_indices = support_valid + support_invalid
        query_indices = query_valid + query_invalid
        random.shuffle(support_indices)
        random.shuffle(query_indices)

        return support_indices, query_indices