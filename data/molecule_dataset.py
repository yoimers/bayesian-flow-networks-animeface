import torch
import pickle
from torch.utils.data import Dataset

class MoleculeDataset(Dataset):
    def __init__(self, data_path, conditioned=False):
        with open(data_path, 'rb') as file:
            self.data = pickle.load(file)    
        self.length = len(self.data)
        self.conditioned = conditioned

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        x = self.data[ind]['emb_smiles']
        y = None
        minX = min(x)
        maxX = max(x)
        x = torch.FloatTensor((2*((x-minX)/(maxX - minX)) - 1)).reshape(1, 16, 16)
        if self.conditioned:
            y = torch.tensor(self.data[ind]['logp'])
        return x, y