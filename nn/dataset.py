from torch.utils.data import Dataset
import torch

class MatricesDataset(Dataset):
    def __init__(self, df, matrices, target_name):
        self.df = df
        self.matrices = matrices
        self.target_name = target_name

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, index):
        matrix = self.matrices[index]
        hodge_number = self.df.iloc[index][self.target_name]
        return {'matrix': torch.tensor(matrix, dtype=torch.float32), 'hodge_number': torch.tensor(hodge_number, dtype=torch.float32)}