from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings['UserID'].values
        self.items = ratings['MovieID'].values
        self.ratings = ratings['Rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'rating': self.ratings[idx]
        }