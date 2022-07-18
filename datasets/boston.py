from sklearn.datasets import load_boston
from torch.utils.data import Dataset

class BostonDataset(Dataset):
    def __init__(self):
        boston = load_boston()
        self.features = boston["data"]
        self.target = boston["target"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feat, target = self.features[index], self.target[index]
        return feat, target