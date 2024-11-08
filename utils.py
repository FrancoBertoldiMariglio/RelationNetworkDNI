import torch
from torch.utils.data import Dataset
from torch import Tensor
import os
from PIL import Image
import random
from typing import List, Tuple, Optional, Dict, Union
from torchvision import transforms

class BinaryImageDataset(Dataset):
    """
    Memory-efficient dataset implementation maintaining data on CPU until loaded by DataLoader
    """
    def __init__(self,
                 valid_dir: str, 
                 invalid_dir: str, 
                 transform: Optional[transforms.Compose] = None) -> None:
        if not os.path.exists(valid_dir) or not os.path.exists(invalid_dir):
            raise ValueError("Invalid directory paths")
            
        self.cache: Dict[int, Image.Image] = {}
        self.cache_size: int = 100
        self.transform = transform

        self.valid_images = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.invalid_images = [os.path.join(invalid_dir, f) for f in os.listdir(invalid_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.valid_images or not self.invalid_images:
            raise ValueError("No valid images found in one or both directories")

        self.all_images = self.valid_images + self.invalid_images
        # Create labels tensor on CPU
        self.labels = torch.tensor([1] * len(self.valid_images) + [0] * len(self.invalid_images), 
                                 dtype=torch.int8)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get item maintained on CPU for DataLoader to handle GPU transfer
        Args:
            idx: Index of the item to retrieve
        Returns:
            Tuple of (image tensor, label tensor) both on CPU
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        if idx in self.cache:
            image = self.cache[idx]
        else:
            try:
                image = Image.open(self.all_images[idx]).convert('RGB')
                if len(self.cache) < self.cache_size:
                    self.cache[idx] = image
            except Exception as e:
                raise RuntimeError(f"Error loading image {self.all_images[idx]}: {str(e)}")

        # Convert label to float32 but keep on CPU
        label = self.labels[idx].to(torch.float32)
        
        if self.transform:
            try:
                image = self.transform(image)
                image = image.contiguous().to('cuda', non_blocking=True)
            except Exception as e:
                raise RuntimeError(f"Error transforming image {self.all_images[idx]}: {str(e)}")

        label = label.to('cuda', non_blocking=True)
        
        return image, label

    def __len__(self) -> int:
        return len(self.all_images)


class EpisodeSampler:
    """
    Episode sampler with CPU-based indexing
    """
    def __init__(self, dataset: BinaryImageDataset, n_shot: int, n_query: int) -> None:
        if not isinstance(dataset, BinaryImageDataset):
            raise TypeError("Dataset must be an instance of BinaryImageDataset")
        if not isinstance(n_shot, int) or not isinstance(n_query, int):
            raise TypeError("n_shot and n_query must be integers")
            
        self.n_shot = n_shot
        self.n_query = n_query
        
        # Keep indices on CPU
        self.valid_indices = torch.where(dataset.labels == 1)[0].tolist()
        self.invalid_indices = torch.where(dataset.labels == 0)[0].tolist()

        if n_shot > min(len(self.valid_indices), len(self.invalid_indices)):
            raise ValueError("n_shot is larger than available samples")
        if n_query > min(len(self.valid_indices) - n_shot, len(self.invalid_indices) - n_shot):
            raise ValueError("n_query is too large given n_shot and available samples")

    def sample_episode(self) -> Tuple[List[int], List[int]]:
        """
        Sample episode indices
        Returns:
            Tuple of (support indices, query indices)
        """
        support_valid = random.sample(self.valid_indices, self.n_shot)
        support_invalid = random.sample(self.invalid_indices, self.n_shot)

        remaining_valid = list(set(self.valid_indices) - set(support_valid))
        remaining_invalid = list(set(self.invalid_indices) - set(support_invalid))

        query_valid = random.sample(remaining_valid, self.n_query)
        query_invalid = random.sample(remaining_invalid, self.n_query)

        support_indices = support_valid + support_invalid
        query_indices = query_valid + query_invalid
        random.shuffle(support_indices)
        random.shuffle(query_indices)

        return support_indices, query_indices