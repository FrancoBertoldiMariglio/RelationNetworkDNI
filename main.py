import argparse
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import math
from PIL import Image
from models.RelationNet import EmbeddingNet, RelationModule


class BinaryImageDataset(Dataset):
    """Custom PyTorch Dataset for valid and invalid images"""
    def __init__(self, valid_dir, invalid_dir, transform=None):
        self.valid_images = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir)]
        self.invalid_images = [os.path.join(invalid_dir, f) for f in os.listdir(invalid_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.valid_images) + len(self.invalid_images)

    def __getitem__(self, idx):
        if idx < len(self.valid_images):
            image = Image.open(self.valid_images[idx]).convert('RGB')
            label = 1
        else:
            image = Image.open(self.invalid_images[idx - len(self.valid_images)]).convert('RGB')
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, label

def weights_init(m):
    """Custom weight initialization function"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main(valid_dir, invalid_dir, epochs, episodes_per_epoch, n_shot, n_query, learning_rate, device):

    # Initialize data dirs
    print("Initializing data dirs...")

    best_eval_loss = float('inf')

    # Initialize neural networks
    print("Initializing neural networks...")
    embedding_net = EmbeddingNet().to(device)
    relation_module = RelationModule().to(device)

    embedding_net.apply(weights_init)
    relation_module.apply(weights_init)

    embedding_optim = torch.optim.Adam(embedding_net.parameters(), lr=learning_rate)
    embedding_scheduler = StepLR(embedding_optim, step_size=10000, gamma=0.5)
    relation_optim = torch.optim.Adam(relation_module.parameters(), lr=learning_rate)
    relation_scheduler = StepLR(relation_optim, step_size=10000, gamma=0.5)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BinaryImageDataset(valid_dir, invalid_dir, transform)

    # Training loop
    print("Training...")
    for epoch in range(epochs):
        for episode in range(episodes_per_epoch):
            # Sample support and query sets
            support_images, support_labels, query_images, query_labels = sample_episode(dataset, n_shot, n_query, device)

            # Extract features
            support_features = embedding_net(support_images.to(device))
            query_features = embedding_net(query_images.to(device))

            # Combine query and support features
            n_query = query_features.size(0)
            n_support = support_features.size(0)

            # Expand dimensions for comparison
            query_features_ext = query_features.unsqueeze(1).repeat(1, n_support, 1, 1, 1)
            support_features_ext = support_features.unsqueeze(0).repeat(n_query, 1, 1, 1, 1)

            # Concatenate query and support features
            relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
            relation_pairs = relation_pairs.view(-1, relation_pairs.size(2),
                                                 relation_pairs.size(3),
                                                 relation_pairs.size(4)).to(device)

            # Compute relations
            relations = relation_module(relation_pairs)
            relations = relations.view(n_query, n_support)

            # Weight relations by support labels
            support_labels = support_labels.float()
            weighted_relations = relations * support_labels

            predictions = weighted_relations.mean(dim=1)

            # Compute loss and optimize
            train_loss = nn.BCELoss()(predictions, query_labels.float().to(device))
            embedding_optim.zero_grad()
            relation_optim.zero_grad()
            train_loss.backward()
            embedding_optim.step()
            relation_optim.step()

            # Update learning rates
            embedding_scheduler.step()
            relation_scheduler.step()

            train_accuracy = (predictions >= 0.5).long().squeeze() == query_labels.long().squeeze().to(device)

            if (episode + 1) % 10 == 0:
                print(f"Epoch: {epoch+1}, Episode: {episode+1}, Loss: {train_loss.item()}")

        # Evaluate on test set
        eval_accuracy, eval_precision, eval_recall, eval_f1, eval_loss = evaluate_model(embedding_net, relation_module, dataset, n_shot, n_query, device)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}, '
              f'Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_accuracy:.2f}')

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(embedding_net.state_dict(), 'models/best_embedding_net.pth')
            torch.save(relation_module.state_dict(), 'models/best_relation_module.pth')
            print("Checkpoint saved")


def sample_episode(dataset, n_shot, n_query, device):
    """Sample support and query sets for an episode"""
    total_len = len(dataset)

    # Check if there are enough samples in the dataset
    if n_shot * 2 > total_len or n_query * 2 > total_len:
        raise ValueError("Not enough samples in the dataset for the specified n_shot and n_query values")

    # Sample support and query sets
    support_indices = random.sample(range(total_len), min(n_shot * 2, total_len))
    query_indices = random.sample(range(total_len), min(n_query * 2, total_len))

    support_images = torch.stack([dataset[i][0] for i in support_indices]).to(device)
    support_labels = torch.tensor([dataset[i][1] for i in support_indices]).to(device)

    query_images = torch.stack([dataset[i][0] for i in query_indices]).to(device)
    query_labels = torch.tensor([dataset[i][1] for i in query_indices]).to(device)

    return support_images, support_labels, query_images, query_labels


def evaluate_model(embedding_net, relation_module, dataset, n_shot, n_query, device):
    """Evaluate the model on the test set"""
    embedding_net.eval()
    relation_module.eval()
    criterion = nn.BCELoss()

    # Load data
    total_len = len(dataset)
    test_indices = random.sample(range(total_len), 100 * (n_shot + n_query))

    test_images = torch.stack([dataset[i][0] for i in test_indices]).to(device)
    test_labels = torch.tensor([dataset[i][1] for i in test_indices]).to(device)

    # Extract features
    test_features = embedding_net(test_images)

    # Compute relations
    relations = relation_module(test_features[:n_query * 100], test_features[n_query * 100:], test_labels[:n_shot * 100].to(device))
    predictions = (relations >= 0.5).long().squeeze()

    # Compute loss
    loss = criterion(relations, test_labels[:n_query * 100].float())
    total_loss = loss.item() * (n_query * 100)

    # Compute accuracy
    total_correct = (predictions == test_labels[:n_query * 100].to(device)).sum().item()
    total_samples = n_query * 100

    # Calculate additional metrics
    accuracy = total_correct / total_samples
    precision = precision_score(test_labels[:n_query * 100].tolist(), predictions.tolist())
    recall = recall_score(test_labels[:n_query * 100].tolist(), predictions.tolist())
    f1 = f1_score(test_labels[:n_query * 100].tolist(), predictions.tolist())
    eval_loss = total_loss / (n_query * 100)

    return accuracy, precision, recall, f1, eval_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_dir', type=str, default='data/valid', help='Directory containing valid images')
    parser.add_argument('--invalid_dir', type=str, default='data/invalid', help='Directory containing invalid images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=100, help='Number of episodes per epoch')
    parser.add_argument('--n_shot', type=int, default=5, help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')

    args = parser.parse_args()

    main(args.valid_dir, args.invalid_dir, args.epochs, args.episodes_per_epoch, args.n_shot, args.n_query, args.learning_rate, args.device)