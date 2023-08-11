import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch import nn

class PairedDataset(Dataset):
    def __init__(self, fn1, fn2):
        self.keys = list(dict1.keys())
        with open(fn1, 'rb') as f:
            self.dict1 = pickle.load(f)
        with open(fn2, 'rb') as f:
            self.dict2 = pickle.load(f)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.dict1[key], self.dict2[key]

     def get_embedding_dim1(self):
        return len(next(iter(self.dict1.values())))

    def get_embedding_dim2(self):
        return len(next(iter(self.dict2.values())))



def main():
    dataset = PairedDataset(dict1, dict2)
    embedding_dim1 = dataset.get_embedding_dim1()
    embedding_dim2 = dataset.get_embedding_dim2()
    latent_dim = 512
    temp = 1.

    encoder1 = nn.Linear(embedding_dim1, latent_dim)
    encoder2 = nn.Linear(embedding_dim2, latent_dim)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=0.001)
    cross_entropy = nn.CrossEntropyLoss()

    for epoch in range(10):  # Number of epochs
        for batch1, batch2 in loader:
            optimizer.zero_grad()

            z1 = encoder1(batch1)
            z2 = encoder2(batch2)

            e1 = torch.linalg.vector_norm(z1, dim=1)
            e2 = torch.linalg.vector_norm(z2, dim=1)
            logits = e1 @ e2 * torch.exp(t)
            labels = torch.arange(len(batch1))
            loss1 = cross_entropy(logits, labels, axis=0)
            loss2 = cross_entropy(logits, labels, axis=1)
            loss = (loss1 + loss2) / 2
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(encoder1, 'encoder1_model.pth')
    torch.save(encoder2, 'encoder2_model.pth')
