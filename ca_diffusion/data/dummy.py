import torch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data_shape=(3,256,256), length=10000):
        super().__init__()

        self.data_shape = list(data_shape)
        self.length = length

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return {"image": torch.randn(self.data_shape)}
