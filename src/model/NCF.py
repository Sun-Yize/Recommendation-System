import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embed_size, layers):
        super(NCF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.movie_embed = nn.Embedding(num_movies, embed_size)
        self.mlp = nn.Sequential()
        last_size = 2 * embed_size
        for layer_size in layers:
            self.mlp.add_module('layer_{}'.format(layer_size), nn.Linear(last_size, layer_size))
            self.mlp.add_module('activation_{}'.format(layer_size), nn.ReLU())
            last_size = layer_size
        self.output = nn.Linear(last_size, 1)

    def forward(self, user_indices, movie_indices):
        user_embedding = self.user_embed(user_indices)
        movie_embedding = self.movie_embed(movie_indices)
        x = torch.cat([user_embedding, movie_embedding], dim=1)
        x = self.mlp(x)
        x = self.output(x)
        return x.squeeze()