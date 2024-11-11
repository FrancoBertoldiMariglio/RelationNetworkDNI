# Import necessary libraries
import base64
import io

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
from typing import List, Dict, Union, Tuple

from models.RelationNet import EmbeddingNet, RelationModule


class RelationNetPredictor:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.embedding_net = EmbeddingNet().to(self.device)
        self.relation_module = RelationModule().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.embedding_net.load_state_dict(checkpoint['embedding_state_dict'])
        self.relation_module.load_state_dict(checkpoint['relation_state_dict'])

        # Set models to evaluation mode
        self.embedding_net.eval()
        self.relation_module.eval()

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def load_and_transform_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and transform a single image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def load_and_transform_image_base_64(self, image_base64: str) -> torch.Tensor:
        """
        Load and transform a base64 encoded image

        Args:
            image_base64: Base64 string of the image (can include or exclude the data URI prefix)

        Returns:
            torch.Tensor: Transformed image tensor
        """
        try:
            # Remove data URI prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Create PIL Image from bytes
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Transform and return
            return self.transform(image).unsqueeze(0).to(self.device)

        except Exception as e:
            raise ValueError(f"Error processing base64 image: {str(e)}")


    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Get embedding for an image"""
        with torch.no_grad():
            return self.embedding_net(image_tensor)

    def predict_single_modified(self, query_image, support_images, support_labels):
        """Versión modificada que da más peso a las coincidencias exactas"""
        # Obtener embeddings
        query_tensor = self.load_and_transform_image(query_image)
        query_features = self.get_embedding(query_tensor)

        support_features = []
        for img_path in support_images:
            img_tensor = self.load_and_transform_image(img_path)
            features = self.get_embedding(img_tensor)
            support_features.append(features)

        support_features = torch.cat(support_features)
        support_labels = torch.tensor(support_labels, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Calcular similitud directa con cada imagen de soporte
            similarities = []
            for i in range(len(support_features)):
                similarity = F.cosine_similarity(
                    query_features.view(1, -1),
                    support_features[i].view(1, -1)
                )
                similarities.append(similarity.item())

            # Si hay una coincidencia casi exacta con una imagen inválida
            max_similarity = max(similarities)
            if max_similarity > 0.95:  # Umbral de similitud alto
                idx = similarities.index(max_similarity)
                if support_labels[idx] == 0:  # Si la imagen más similar es inválida
                    return 0.0  # Forzar predicción como inválida

        # Si no hay coincidencia exacta, usar el método normal
        n_support = support_features.size(0)
        query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
        support_features_ext = support_features.unsqueeze(0).expand(1, -1, -1, -1, -1)

        relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
        relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

        relations = self.relation_module(relation_pairs)
        relations = relations.view(1, n_support)

        weighted_relations = relations * support_labels
        prediction = weighted_relations.mean(dim=1)
        return torch.sigmoid(prediction).item()




    # def predict_single(self,
    #                    query_image: Union[str, Path],
    #                    support_images: List[Union[str, Path]],
    #                    support_labels: List[int]) -> float:
    #     """
    #     Predict for a single query image using provided support set
    #
    #     Args:
    #         query_image: Path to query image
    #         support_images: List of paths to support images
    #         support_labels: Binary labels for support images (0 or 1)
    #
    #     Returns:
    #         Dictionary containing prediction probability and binary prediction
    #     """
    #     # Transform query image
    #     query_tensor = self.load_and_transform_image(query_image)
    #     query_features = self.get_embedding(query_tensor)
    #
    #     # Process support set
    #     support_features = []
    #     for img_path in support_images:
    #         img_tensor = self.load_and_transform_image(img_path)
    #         features = self.get_embedding(img_tensor)
    #         support_features.append(features)
    #
    #     support_features = torch.cat(support_features)
    #     support_labels = torch.tensor(support_labels, dtype=torch.float32).to(self.device)
    #
    #     # Compute relations
    #     with torch.no_grad():
    #         n_support = support_features.size(0)
    #         query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
    #         support_features_ext = support_features.unsqueeze(0).expand(1, -1, -1, -1, -1)
    #
    #         relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
    #         relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])
    #
    #         relations = self.relation_module(relation_pairs)
    #         relations = relations.view(1, n_support)
    #
    #         weighted_relations = relations * support_labels
    #         prediction = weighted_relations.mean(dim=1)
    #         probability = torch.sigmoid(prediction).item()
    #
    #     return probability
    #
    #
    # def predict_batch(self,
    #                   query_images: List[Union[str, Path]],
    #                   support_images: List[Union[str, Path]],
    #                   support_labels: List[int]) -> List[float]:
    #     """Batch prediction version"""
    #     results = []
    #     for query_image in query_images:
    #         result = self.predict_single(query_image, support_images, support_labels)
    #         results.append(result)
    #     return results
    #
    def predict_batch_modified(self,
                      query_images: List[Union[str, Path]],
                      support_images: List[Union[str, Path]],
                      support_labels: List[int]) -> List[float]:
        """Batch prediction version"""
        results = []
        for query_image in query_images:
            result = self.predict_single_modified(query_image, support_images, support_labels)
            results.append(result)
        return results
