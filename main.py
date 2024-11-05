import argparse
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from dataloader import ImageDataset
from models.RelationNet import EmbeddingNet, RelationNetwork, weights_init


def main(epochs, episodes_per_epoch, n_shot, n_query, learning_rate, device):

    # Initialize data folders
    print("Initializing data folders...")
    valid_folder = 'data/valid'
    invalid_folder = 'data/invalid'

    # Initialize neural networks
    print("Initializing neural networks...")
    embedding_net = EmbeddingNet().to(device)
    relation_network = RelationNetwork().to(device)

    embedding_net.apply(weights_init)
    relation_network.apply(weights_init)

    embedding_optim = torch.optim.Adam(embedding_net.parameters(), lr=learning_rate)
    embedding_scheduler = StepLR(embedding_optim, step_size=10000, gamma=0.5)
    relation_optim = torch.optim.Adam(relation_network.parameters(), lr=learning_rate)
    relation_scheduler = StepLR(relation_optim, step_size=10000, gamma=0.5)

    # Training loop
    print("Training...")
    for epoch in range(epochs):
        for episode in range(episodes_per_epoch):
            # Sample support and query sets
            support_images, support_labels, query_images, query_labels = sample_episode(
                valid_folder, invalid_folder, n_shot, n_query)

            # Extract features
            support_features = embedding_net(support_images.to(device))
            query_features = embedding_net(query_images.to(device))

            # Compute relations
            relations = relation_network(query_features, support_features, support_labels.to(device))

            # Compute loss and optimize
            loss = nn.BCELoss()(relations, query_labels.to(device).float())
            embedding_optim.zero_grad()
            relation_optim.zero_grad()
            loss.backward()
            embedding_optim.step()
            relation_optim.step()

            # Update learning rates
            embedding_scheduler.step()
            relation_scheduler.step()

            if (episode + 1) % 10 == 0:
                print(f"Epoch: {epoch+1}, Episode: {episode+1}, Loss: {loss.item()}")

        # Evaluate on test set
        accuracy, precision, recall, f1, eval_loss = evaluate_model(embedding_net, relation_network, valid_folder, invalid_folder, n_shot, n_query, device)
        print(f"Epoch {epoch+1} Test Accuracy: {test_accuracy:.2f}")

        # Save model checkpoints
        torch.save(embedding_net.state_dict(), f'models/embedding_net_epoch_{epoch+1}.pth')
        torch.save(relation_network.state_dict(), f'models/relation_network_epoch_{epoch+1}.pth')

def sample_episode(valid_folder, invalid_folder, n_shot, n_query):
    """Sample support and query sets for an episode"""
    # Load data
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_dataset = ImageDataset(valid_folder, transform)
    invalid_dataset = ImageDataset(invalid_folder, transform)

    # Sample support and query sets
    valid_indices = random.sample(range(len(valid_dataset)), n_shot + n_query)
    invalid_indices = random.sample(range(len(invalid_dataset)), n_shot + n_query)

    support_images = torch.cat([valid_dataset[i][0].unsqueeze(0) for i in valid_indices[:n_shot]] +
                              [invalid_dataset[i][0].unsqueeze(0) for i in invalid_indices[:n_shot]], dim=0)
    support_labels = torch.cat([torch.tensor([1]) for _ in range(n_shot)] +
                              [torch.tensor([0]) for _ in range(n_shot)], dim=0)

    query_images = torch.cat([valid_dataset[i][0].unsqueeze(0) for i in valid_indices[n_shot:]] +
                            [invalid_dataset[i][0].unsqueeze(0) for i in invalid_indices[n_shot:]], dim=0)
    query_labels = torch.cat([torch.tensor([1]) for _ in range(n_query)] +
                            [torch.tensor([0]) for _ in range(n_query)], dim=0)

    return support_images, support_labels, query_images, query_labels

def evaluate_model(embedding_net, relation_network, valid_folder, invalid_folder, n_shot, n_query, device):
    """Evaluate the model on the test set"""
    embedding_net.eval()
    relation_network.eval()
    criterion = nn.BCELoss()

    # Load data
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_dataset = ImageDataset(valid_folder, transform)
    invalid_dataset = ImageDataset(invalid_folder, transform)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    for _ in range(100):  # Test on 100 episodes
        # Sample support and query sets
        support_images, support_labels, query_images, query_labels = sample_episode(
            valid_folder, invalid_folder, n_shot, n_query)

        # Extract features
        support_features = embedding_net(support_images.to(device))
        query_features = embedding_net(query_images.to(device))

        # Compute relations
        relations = relation_network(query_features, support_features, support_labels.to(device))
        predictions = (relations >= 0.5).long().squeeze()

        # Compute loss
        loss = criterion(relations, query_labels.to(device).float())
        total_loss += loss.item()

        # Compute accuracy
        total_correct += (predictions == query_labels.to(device)).sum().item()
        total_samples += query_labels.size(0)

        # Collect predictions and labels for other metrics
        all_predictions.extend(predictions.tolist())
        all_labels.extend(query_labels.tolist())

    # Calculate additional metrics
    accuracy = total_correct / total_samples
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    eval_loss = total_loss / 100

    return accuracy, precision, recall, f1, eval_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=100, help='Number of episodes per epoch')
    parser.add_argument('--n_shot', type=int, default=5, help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')

    args = parser.parse_args()

    main(args.epochs, args.episodes_per_epoch, args.n_shot, args.n_query, args.learning_rate, args.device)