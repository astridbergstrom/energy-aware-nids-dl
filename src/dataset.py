import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset wrapper for NSL-KDD features and labels
class NSLKDDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create PyTorch DataLoaders for training and testing
def create_dataloaders(train_X, train_y, test_X, test_y, batch_size=32):
        train_dataset = NSLKDDDataset(train_X, train_y)
        test_dataset = NSLKDDDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader