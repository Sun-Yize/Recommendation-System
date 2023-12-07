import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, features):
        support = self.linear(features)
        output = torch.spmm(adjacency_matrix, support)
        return output

class GCN(nn.Module):
    def __init__(self, num_users, num_items, in_features, hidden_size=64):
        super(GCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, in_features)
        self.item_embedding = nn.Embedding(num_items, in_features)
        self.gcn1 = GCNLayer(in_features, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, user_indices, item_indices, adjacency_matrix):
        num_users, num_items = self.user_embedding.num_embeddings, self.item_embedding.num_embeddings
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        features = torch.zeros(num_users + num_items, user_embedded.size(1))
        features.scatter_(0, user_indices.unsqueeze(1).repeat(1, user_embedded.size(1)), user_embedded)
        features.scatter_(0, (num_users + item_indices).unsqueeze(1).repeat(1, item_embedded.size(1)), item_embedded)

        hidden = F.relu(self.gcn1(adjacency_matrix, features))
        output = F.relu(self.gcn2(adjacency_matrix, hidden))
        user_output, item_output = output.split([num_users, num_items], 0)
        user_output = user_output[user_indices]
        item_output = item_output[item_indices]
        interaction = torch.mul(user_output, item_output)
        rating = self.fc(interaction)
        return rating.squeeze()