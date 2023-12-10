import torch
import torch.nn as nn
import random

class GCN(nn.Module):
    def __init__(self, num_ent, args, device):
        super(GCN, self).__init__()
        self.num_ent = num_ent
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.device = device

        self._gen_adj()
        self.ent = torch.nn.Embedding(num_ent, args.dim)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        # Your adjacency matrix generation logic here

    def forward(self, v):
        '''
        input: v is batch sized indices for nodes
        v: [batch_size]
        '''
        batch_size = v.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        v = v.view((-1, 1))
        
        # [batch_size, dim]
        entity_embeddings = self.ent(v).squeeze(dim=1)
        entities = self._get_neighbors(v)

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self._aggregate(entity_embeddings, entities[hop])
                entity_vectors_next_iter.append(vector)
            entity_embeddings = entity_vectors_next_iter[0].view((self.batch_size, self.dim))

        return entity_embeddings

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for nodes
        v: [batch_size, 1]
        '''
        entities = [v]
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            
        return entities
    
    def _aggregate(self, entity_embeddings, neighbor_entities):
        '''
        Aggregate neighbor vectors
        '''
        neighbor_vectors = self.ent(neighbor_entities).view((self.batch_size, -1, self.n_neighbor, self.dim))
        aggregated_vector = torch.mean(neighbor_vectors, dim=2)  # Mean aggregation
        return aggregated_vector
