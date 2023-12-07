import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50, hidden_layers=[100, 50], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        layers = []
        input_size = embedding_size * 2  # User and item embeddings concatenated
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = layer_size

        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        concatenated = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.hidden_layers(concatenated)
        return x.squeeze()
