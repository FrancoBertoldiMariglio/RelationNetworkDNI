import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class EmbeddingNet(nn.Module):
    """CNN for feature extraction from images"""

    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationModule(nn.Module):
    """Module to compute relations between query and support examples"""

    def __init__(self, input_size=64, hidden_size=8):
        super(RelationModule, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


class RelationNetwork(nn.Module):
    def __init__(self, input_channels=3):
        super(RelationNetwork, self).__init__()
        self.embedding = EmbeddingNet()
        self.relation = RelationModule()

    def forward(self, queries, supports, support_labels):
        # Extract features
        support_features = self.embedding(supports)  # [n_support, 64, 8, 8]
        query_features = self.embedding(queries)  # [n_query, 64, 8, 8]

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
                                             relation_pairs.size(4))

        # Compute relations
        relations = self.relation(relation_pairs)
        relations = relations.view(n_query, n_support)

        # Weight relations by support labels
        support_labels = support_labels.float()
        weighted_relations = relations * support_labels

        # Average relations for final prediction
        predictions = weighted_relations.mean(dim=1)
        return predictions