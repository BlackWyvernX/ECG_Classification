import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from data_prepros import gendataset, load_data

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.X = torch.tensor(signals, dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.Y[idx]

def create_dataset():
    X, Y = load_data()
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)

    return train_dataset, test_dataset, weights

