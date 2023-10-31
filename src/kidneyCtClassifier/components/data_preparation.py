import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from kidneyCtClassifier.entity.config_entity import TrainingConfig
from kidneyCtClassifier.utils.common import transform_data
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class DataPreparation:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = transform_data()

    def load_data(self):
        dataset = ImageFolder(str(self.config.training_data), transform=self.transform['train'])
        datasets = self.train_val_dataset(dataset)
        # print(len(datasets['train']))
        # print(len(datasets['val']))
        # The original dataset is available in the Subset class
        # print(datasets['train'].dataset)
        dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=self.config.params_batch_size, pin_memory=True,
                                     shuffle=False) for x in ['train', 'val']}

        return dataloaders

    @staticmethod
    def train_val_dataset(dataset, val_split=0.20):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets
