import torch
from torch.utils.data import Dataset

class LinearDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)

        self.target = torch.tensor([4.185, 7.139, 9.678, 12.901, 15.511, 19.347, 22.057, 25.148, 28.296, 31.587], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feat, target = self.features[index], self.target[index]
        return feat, target