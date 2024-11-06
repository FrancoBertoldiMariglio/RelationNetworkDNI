import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from datetime import datetime
from models.RelationNet import EmbeddingNet, RelationModule
from tqdm import tqdm

from utils import BinaryImageDataset, EpisodeSampler


class RelationNetTrainer:
    def __init__(self, config: Dict):
        self.scaler = None
        self.val_dataset = None
        self.train_dataset = None
        self.val_transform = None
        self.train_transform = None
        self.relation_scheduler = None
        self.embedding_scheduler = None
        self.relation_optim = None
        self.embedding_optim = None
        self.relation_module = None
        self.embedding_net = None
        self.logger = None
        self.config = config
        self.device = torch.device(config['device'])
        self.best_val_loss = float('inf')

        # Setup logging
        self.setup_logging()

        # Initialize models and move to GPU
        self.setup_models()

        # Initialize optimizers and schedulers
        self.setup_optimizers()

        # Initialize criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup data
        self.setup_data()

        # Initialize gradient scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda')

    def setup_logging(self):
        """Initialize logging configuration"""
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
        self.logger = logging.getLogger(__name__)

    def setup_models(self):
        """Initialize and configure models"""
        self.embedding_net = EmbeddingNet().to(self.device)
        self.relation_module = RelationModule().to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.embedding_net = nn.DataParallel(self.embedding_net)
            self.relation_module = nn.DataParallel(self.relation_module)

    def setup_optimizers(self):
        """Initialize optimizers and schedulers"""
        self.embedding_optim = optim.AdamW(
            self.embedding_net.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        self.relation_optim = optim.AdamW(
            self.relation_module.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )

        total_steps = self.config['epochs'] * self.config['episodes_per_epoch']

        self.embedding_scheduler = CosineAnnealingLR(
            self.embedding_optim,
            T_max=total_steps,
            eta_min=1e-6
        )
        self.relation_scheduler = CosineAnnealingLR(
            self.relation_optim,
            T_max=total_steps,
            eta_min=1e-6
        )

    def setup_data(self):
        """Initialize datasets and data transforms"""
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Simpler transform for validation
        self.val_transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize datasets
        self.train_dataset = BinaryImageDataset(
            self.config['valid_dir'],
            self.config['invalid_dir'],
            transform=self.train_transform
        )

        self.val_dataset = BinaryImageDataset(
            self.config['valid_dir'],
            self.config['invalid_dir'],
            transform=self.val_transform
        )

    def train_episode(self, support_loader: DataLoader, query_loader: DataLoader) -> Tuple[float, float]:
        """Train on a single episode"""
        self.embedding_net.train()
        self.relation_module.train()

        # Process support set
        support_features = []
        support_labels = []

        with torch.amp.autocast('cuda'):
            for images, labels in support_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                features = self.embedding_net(images)
                support_features.append(features)
                support_labels.append(labels)

            support_features = torch.cat(support_features)
            support_labels = torch.cat(support_labels)

            # Process query set
            query_features = []
            query_labels = []

            for images, labels in query_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                features = self.embedding_net(images)
                query_features.append(features)
                query_labels.append(labels)

            query_features = torch.cat(query_features)
            query_labels = torch.cat(query_labels)

            # Compute relations efficiently
            n_query = query_features.size(0)
            n_support = support_features.size(0)

            query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
            support_features_ext = support_features.unsqueeze(0).expand(n_query, -1, -1, -1, -1)

            relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
            relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

            relations = self.relation_module(relation_pairs)
            relations = relations.view(n_query, n_support)

            # Weight relations by support labels
            weighted_relations = relations * support_labels.float()
            predictions = weighted_relations.mean(dim=1)

            # Use scaler for loss computation
            loss = self.criterion(predictions, query_labels.float())

        # Optimize with gradient scaling
        self.embedding_optim.zero_grad()
        self.relation_optim.zero_grad()

        self.scaler.scale(loss).backward()

        # Gradient clipping with scaler
        self.scaler.unscale_(self.embedding_optim)
        self.scaler.unscale_(self.relation_optim)
        torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.relation_module.parameters(), 1.0)

        self.scaler.step(self.embedding_optim)
        self.scaler.step(self.relation_optim)
        self.scaler.update()

        # Update learning rates
        self.embedding_scheduler.step()
        self.relation_scheduler.step()

        # Apply sigmoid for accuracy computation since we're using BCEWithLogitsLoss
        with torch.no_grad():
            accuracy = ((torch.sigmoid(predictions) >= 0.5).long() == query_labels).float().mean()

        return loss.item(), accuracy.item()

    def _evaluate_episode(self, support_loader: DataLoader, query_loader: DataLoader) -> Tuple[float, list, list]:
        """Evaluate a single episode"""
        self.embedding_net.eval()
        self.relation_module.eval()

        predictions_list = []
        labels_list = []

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Process support set
                support_features = []
                support_labels = []

                for images, labels in support_loader:
                    images = images.to(self.device, non_blocking=True)  # Asynchronous transfer
                    labels = labels.to(self.device, non_blocking=True)
                    features = self.embedding_net(images)
                    support_features.append(features)
                    support_labels.append(labels)

                support_features = torch.cat(support_features)
                support_labels = torch.cat(support_labels)

                # Process query set
                query_features = []
                query_labels = []

                for images, labels in query_loader:
                    images = images.to(self.device, non_blocking=True)  # Asynchronous transfer
                    labels = labels.to(self.device, non_blocking=True)
                    features = self.embedding_net(images)
                    query_features.append(features)
                    query_labels.append(labels)

                query_features = torch.cat(query_features)
                query_labels = torch.cat(query_labels)

                # Compute relations
                n_query = query_features.size(0)
                n_support = support_features.size(0)

                query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
                support_features_ext = support_features.unsqueeze(0).expand(n_query, -1, -1, -1, -1)

                relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
                relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

                relations = self.relation_module(relation_pairs)
                relations = relations.view(n_query, n_support)

                weighted_relations = relations * support_labels.float()
                predictions = weighted_relations.mean(dim=1)

                loss = self.criterion(predictions, query_labels.float())

                # Apply sigmoid since we're using BCEWithLogitsLoss
                predictions = torch.sigmoid(predictions)
                binary_predictions = (predictions >= 0.5).long()

                predictions_list.extend(binary_predictions.cpu().numpy())
                labels_list.extend(query_labels.cpu().numpy())

        return loss.item(), predictions_list, labels_list

    def evaluate(self, dataset: Optional[BinaryImageDataset] = None) -> Dict[str, float]:
        """Evaluate the model"""
        if dataset is None:
            dataset = self.val_dataset

        episode_sampler = EpisodeSampler(
            dataset,
            self.config['n_shot'],
            self.config['n_query']
        )

        metrics = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

        n_eval_episodes = 10
        all_predictions = []
        all_labels = []

        for _ in range(n_eval_episodes):
            support_indices, query_indices = episode_sampler.sample_episode()

            support_loader = DataLoader(
                Subset(dataset, support_indices),
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=True,
                persistent_workers=True
            )

            query_loader = DataLoader(
                Subset(dataset, query_indices),
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=True,
                persistent_workers=True
            )

            loss, predictions, labels = self._evaluate_episode(
                support_loader,
                query_loader
            )

            metrics['loss'] += loss
            all_predictions.extend(predictions)
            all_labels.extend(labels)

        # Calculate final metrics
        metrics['loss'] /= n_eval_episodes
        metrics['accuracy'] = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        metrics['precision'] = precision_score(all_labels, all_predictions)
        metrics['recall'] = recall_score(all_labels, all_predictions)
        metrics['f1'] = f1_score(all_labels, all_predictions)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'embedding_state_dict': self.embedding_net.state_dict(),
            'relation_state_dict': self.relation_module.state_dict(),
            'embedding_optim_state_dict': self.embedding_optim.state_dict(),
            'relation_optim_state_dict': self.relation_optim.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            # Save best metrics
            with open(checkpoint_dir / 'best_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)


    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.embedding_net.load_state_dict(checkpoint['embedding_state_dict'])
        self.relation_module.load_state_dict(checkpoint['relation_state_dict'])
        self.embedding_optim.load_state_dict(checkpoint['embedding_optim_state_dict'])
        self.relation_optim.load_state_dict(checkpoint['relation_optim_state_dict'])

        return checkpoint['epoch']

    def train(self):
        """Main training loop"""
        episode_sampler = EpisodeSampler(
            self.train_dataset,
            self.config['n_shot'],
            self.config['n_query']
        )

        early_stopping_counter = 0
        early_stopping_patience = 10

        self.logger.info("Starting training...")
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            epoch_accuracy = 0

            # Training loop
            progress_bar = tqdm(range(self.config['episodes_per_epoch']),
                                desc=f"Epoch {epoch + 1}")

            for _ in progress_bar:
                # Sample episode
                support_indices, query_indices = episode_sampler.sample_episode()

                # Create dataloaders
                support_loader = DataLoader(
                    Subset(self.train_dataset, support_indices),
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    pin_memory=True
                )
                query_loader = DataLoader(
                    Subset(self.train_dataset, query_indices),
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    pin_memory=True
                )

                # Train on episode
                loss, accuracy = self.train_episode(support_loader, query_loader)

                epoch_loss += loss
                epoch_accuracy += accuracy

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{accuracy:.4f}'
                })

            # Compute epoch metrics
            epoch_loss /= self.config['episodes_per_epoch']
            epoch_accuracy /= self.config['episodes_per_epoch']

            # Evaluate on validation set
            val_metrics = self.evaluate()

            # Log metrics
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config['epochs']}] "
                f"Train Loss: {epoch_loss:.4f} "
                f"Train Acc: {epoch_accuracy:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Acc: {val_metrics['accuracy']:.4f} "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Check if this is the best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                early_stopping_counter = 0
                self.logger.info("New best model saved!")
            else:
                early_stopping_counter += 1

            # Regular checkpoint saving
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                break

            # Learning rate scheduling logging
            current_lr_embed = self.embedding_optim.param_groups[0]['lr']
            current_lr_rel = self.relation_optim.param_groups[0]['lr']
            self.logger.info(f"Current LR - Embedding: {current_lr_embed:.6f}, "
                             f"Relation: {current_lr_rel:.6f}")

        self.logger.info("Training completed!")

        # Final evaluation on test set if provided
        if hasattr(self, 'test_dataset'):
            self.logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(self.test_dataset)
            self.logger.info(
                f"Final Test Metrics:\n"
                f"Loss: {test_metrics['loss']:.4f}\n"
                f"Accuracy: {test_metrics['accuracy']:.4f}\n"
                f"Precision: {test_metrics['precision']:.4f}\n"
                f"Recall: {test_metrics['recall']:.4f}\n"
                f"F1 Score: {test_metrics['f1']:.4f}"
            )


def main():
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
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Convert args to config dictionary
    config = vars(args)

    # Initialize trainer
    trainer = RelationNetTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        trainer.logger.info(f"Resumed training from epoch {start_epoch}")

    # Print configuration
    trainer.logger.info("Configuration:")
    for key, value in config.items():
        trainer.logger.info(f"{key}: {value}")

    # Print model architecture
    trainer.logger.info("\nModel Architecture:")
    trainer.logger.info("\nEmbedding Network:")
    trainer.logger.info(trainer.embedding_net)
    trainer.logger.info("\nRelation Module:")
    trainer.logger.info(trainer.relation_module)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        trainer.logger.exception("An error occurred during training")
        raise e
    finally:
        # Save final checkpoint
        trainer.logger.info("Saving final checkpoint...")
        trainer.save_checkpoint(
            config['epochs'],
            {'loss': float('inf')},
            is_best=False
        )

if __name__ == '__main__':
    main()