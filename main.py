import argparse
import gc
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.multiprocessing as mp
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
import logging
from tqdm import tqdm

from models.RelationNet import EmbeddingNet, RelationModule
from utils import BinaryImageDataset, EpisodeSampler

# Configura el método de inicio en 'spawn' para evitar problemas de inicialización de CUDA
mp.set_start_method('spawn', force=True)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config: Dict[str, Any] = {}
best_val_loss: float = float('inf')

# Model components
embedding_net: Optional[EmbeddingNet] = None
relation_module: Optional[RelationModule] = None
embedding_optim: Optional[optim.AdamW] = None
relation_optim: Optional[optim.AdamW] = None
criterion: Optional[nn.BCEWithLogitsLoss] = None
scaler: Optional[GradScaler] = None

# Datasets and transforms
train_dataset: Optional[BinaryImageDataset] = None
val_dataset: Optional[BinaryImageDataset] = None
train_transform: Optional[transforms.Compose] = None
val_transform: Optional[transforms.Compose] = None

# Logger
logger = None


def setup_logging() -> None:
    """Initialize logging configuration"""
    global logger

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)


def setup_models() -> None:
    """Initialize models directly on GPU"""
    global embedding_net, relation_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This implementation requires GPU.")

    torch.cuda.empty_cache()

    embedding_net = EmbeddingNet().to(device)
    relation_module = RelationModule().to(device)

    assert next(embedding_net.parameters()).is_cuda
    assert next(relation_module.parameters()).is_cuda


def setup_optimizers() -> None:
    """Initialize optimizers with GPU support"""
    global embedding_optim, relation_optim, criterion, scaler

    embedding_optim = optim.AdamW(
        embedding_net.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    relation_optim = optim.AdamW(
        relation_module.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    criterion = nn.BCEWithLogitsLoss().to(device)

    scaler = GradScaler()


def setup_data() -> None:
    """Initialize datasets and transforms"""
    global train_transform, val_transform, train_dataset, val_dataset

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BinaryImageDataset(
        config['valid_dir'],
        config['invalid_dir'],
        transform=train_transform
    )

    val_dataset = BinaryImageDataset(
        config['valid_dir'],
        config['invalid_dir'],
        transform=val_transform
    )


def train_episode(support_loader: DataLoader, query_loader: DataLoader) -> Tuple[float, float]:
    """Train a single episode with GPU acceleration"""
    embedding_net.train()
    relation_module.train()

    assert next(embedding_net.parameters()).is_cuda, "embedding_net not on CUDA"
    assert next(relation_module.parameters()).is_cuda, "relation_module not on CUDA"

    support_features: List[Tensor] = []
    support_labels: List[Tensor] = []

    with autocast('cuda'):
        torch.cuda.synchronize()
        for images, labels in support_loader:
            if not images.is_cuda:
                images = images.to(device, non_blocking=True)
            if not labels.is_cuda:
                labels = labels.to(device, non_blocking=True)

            features = embedding_net(images)
            support_features.append(features)
            support_labels.append(labels)

        support_features = torch.cat(support_features)
        support_labels = torch.cat(support_labels)

        query_features: List[Tensor] = []
        query_labels: List[Tensor] = []

        for images, labels in query_loader:
            if not images.is_cuda:
                images = images.to(device, non_blocking=True)
            if not labels.is_cuda:
                labels = labels.to(device, non_blocking=True)

            features = embedding_net(images)
            query_features.append(features)
            query_labels.append(labels)

        query_features = torch.cat(query_features)
        query_labels = torch.cat(query_labels)

        n_query = query_features.size(0)
        n_support = support_features.size(0)

        query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
        support_features_ext = support_features.unsqueeze(0).expand(n_query, -1, -1, -1, -1)

        relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
        relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

        relations = relation_module(relation_pairs)
        relations = relations.view(n_query, n_support)

        weighted_relations = relations * support_labels.float()
        predictions = weighted_relations.mean(dim=1)

        loss = criterion(predictions, query_labels.float())

    embedding_optim.zero_grad(set_to_none=True)
    relation_optim.zero_grad(set_to_none=True)

    scaler.scale(loss).backward()

    scaler.unscale_(embedding_optim)
    scaler.unscale_(relation_optim)

    torch.nn.utils.clip_grad_norm_(embedding_net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(relation_module.parameters(), 1.0)

    scaler.step(embedding_optim)
    scaler.step(relation_optim)
    scaler.update()

    with torch.no_grad():
        accuracy = ((torch.sigmoid(predictions) >= 0.5).long() == query_labels).float().mean()

    torch.cuda.synchronize()

    return loss.item(), accuracy.item()

def train() -> None:
    """Main training loop"""
    global best_val_loss

    episode_sampler = EpisodeSampler(
        train_dataset,
        config['n_shot'],
        config['n_query']
    )

    early_stopping_counter = 0
    early_stopping_patience = 10

    logger.info("Starting training...")
    for epoch in range(config['epochs']):
        epoch_loss: float = 0.0
        epoch_accuracy: float = 0.0

        progress_bar = tqdm(range(config['episodes_per_epoch']),
                            desc=f"Epoch {epoch + 1}")

        for _ in progress_bar:
            torch.cuda.empty_cache()
            gc.collect()

            support_indices, query_indices = episode_sampler.sample_episode()

            support_loader = DataLoader(
                Subset(train_dataset, support_indices),
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                persistent_workers=True,
                prefetch_factor=config['prefetch_factor']
            )
            query_loader = DataLoader(
                Subset(train_dataset, query_indices),
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                persistent_workers=True,
                prefetch_factor=config['prefetch_factor']
            )

            loss, accuracy = train_episode(support_loader, query_loader)

            epoch_loss += loss
            epoch_accuracy += accuracy

            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{accuracy:.4f}'
            })

            del loss, accuracy
            torch.cuda.synchronize()

        epoch_loss /= config['episodes_per_epoch']
        epoch_accuracy /= config['episodes_per_epoch']

        val_metrics = evaluate()

        logger.info(
            f"Epoch [{epoch + 1}/{config['epochs']}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"Train Acc: {epoch_accuracy:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(epoch, val_metrics, is_best=True)
            early_stopping_counter = 0
            logger.info("New best model saved!")
        else:
            early_stopping_counter += 1

        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(epoch, val_metrics)

        if early_stopping_counter >= early_stopping_patience:
            logger.info("Early stopping triggered!")
            break

    logger.info("Training completed!")


def _evaluate_episode(support_loader: DataLoader, query_loader: DataLoader) -> Tuple[float, List[int], List[int]]:
    """Evaluate a single episode"""
    embedding_net.eval()
    relation_module.eval()

    predictions_list = []
    labels_list = []

    with torch.no_grad(), autocast('cuda'):
        support_features: List[Tensor] = []
        support_labels: List[Tensor] = []

        for images, labels in support_loader:
            if not images.is_cuda:
                images = images.to(device, non_blocking=True)
            if not labels.is_cuda:
                labels = labels.to(device, non_blocking=True)

            features = embedding_net(images)
            support_features.append(features)
            support_labels.append(labels)

        support_features = torch.cat(support_features)
        support_labels = torch.cat(support_labels)

        query_features: List[Tensor] = []
        query_labels: List[Tensor] = []

        for images, labels in query_loader:
            if not images.is_cuda:
                images = images.to(device, non_blocking=True)
            if not labels.is_cuda:
                labels = labels.to(device, non_blocking=True)

            features = embedding_net(images)
            query_features.append(features)
            query_labels.append(labels)

        query_features = torch.cat(query_features)
        query_labels = torch.cat(query_labels)

        n_query = query_features.size(0)
        n_support = support_features.size(0)

        query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
        support_features_ext = support_features.unsqueeze(0).expand(n_query, -1, -1, -1, -1)

        relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
        relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

        relations = relation_module(relation_pairs)
        relations = relations.view(n_query, n_support)

        weighted_relations = relations * support_labels.float()
        predictions = weighted_relations.mean(dim=1)

        loss = criterion(predictions, query_labels.float())

        predictions = torch.sigmoid(predictions)
        binary_predictions = (predictions >= 0.5).long()

        predictions_list.extend(binary_predictions.cpu().numpy())
        labels_list.extend(query_labels.cpu().numpy())

    return loss.item(), predictions_list, labels_list


def evaluate(dataset: Optional[BinaryImageDataset] = None) -> Dict[str, float]:
    """Evaluate the model"""
    if dataset is None:
        dataset = val_dataset

    episode_sampler = EpisodeSampler(
        dataset,
        config['n_shot'],
        config['n_query']
    )

    metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }

    n_eval_episodes = 10
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for _ in range(n_eval_episodes):
        support_indices, query_indices = episode_sampler.sample_episode()

        support_loader = DataLoader(
            Subset(dataset, support_indices),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        query_loader = DataLoader(
            Subset(dataset, query_indices),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        loss, predictions, labels = _evaluate_episode(support_loader, query_loader)

        metrics['loss'] += loss
        all_predictions.extend(predictions)
        all_labels.extend(labels)

    metrics['loss'] /= n_eval_episodes
    metrics['accuracy'] = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    metrics['precision'] = precision_score(all_labels, all_predictions)
    metrics['recall'] = recall_score(all_labels, all_predictions)
    metrics['f1'] = f1_score(all_labels, all_predictions)

    return metrics


def save_checkpoint(epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
    """Save model checkpoint"""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'embedding_state_dict': embedding_net.state_dict(),
        'relation_state_dict': relation_module.state_dict(),
        'embedding_optim_state_dict': embedding_optim.state_dict(),
        'relation_optim_state_dict': relation_optim.state_dict(),
        'metrics': metrics,
        'config': config
    }

    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        with open(checkpoint_dir / 'best_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)


def load_checkpoint(checkpoint_path: str) -> int:
    """Load model checkpoint"""
    global embedding_net, relation_module, embedding_optim, relation_optim

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedding_net.load_state_dict(checkpoint['embedding_state_dict'])
    relation_module.load_state_dict(checkpoint['relation_state_dict'])
    embedding_optim.load_state_dict(checkpoint['embedding_optim_state_dict'])
    relation_optim.load_state_dict(checkpoint['relation_optim_state_dict'])

    return checkpoint['epoch']


def main() -> None:
    """Main function to setup and run training"""
    global config

    parser = argparse.ArgumentParser(description='RelationNet Training Script')

    # Data parameters
    parser.add_argument('--valid_dir', type=str, required=True,
                        help='Directory containing valid images')
    parser.add_argument('--invalid_dir', type=str, required=True,
                        help='Directory containing invalid images')
    parser.add_argument('--test_dir_valid', type=str,
                        help='Directory containing test valid images')
    parser.add_argument('--test_dir_invalid', type=str,
                        help='Directory containing test invalid images')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--episodes_per_epoch', type=int, default=100,
                        help='Number of episodes per epoch')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15,
                        help='Number of query examples per class')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=8,
                        help='Hidden size for relation module')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Runtime parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This implementation requires GPU.")

    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Setup config
    config = vars(args)
    config['device'] = 'cuda'

    # Initialize all components
    setup_logging()
    setup_models()
    setup_optimizers()
    setup_data()

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = load_checkpoint(args.resume)
        logger.info(f"Resumed training from epoch {start_epoch}")

    logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # Start training
    try:
        train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("An error occurred during training")
        raise e
    finally:
        logger.info("Saving final checkpoint...")
        save_checkpoint(
            config['epochs'],
            {'loss': float('inf')},
            is_best=False
        )
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()