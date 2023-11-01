import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from kidneyCtClassifier.components.data_preparation import DataPreparation
from kidneyCtClassifier.config.configuration import ConfigurationManager
from kidneyCtClassifier.entity.config_entity import TrainingConfig, PrepareBaseModelConfig

# Training parameters
# torch.cuda.set_device(0)
torch.manual_seed(1)


class ModelTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, train_loader):

        # Load a cuda pre-trained model and
        # Specify that the model should be loaded on the 'cpu' device, instead of the 'cuda' device.
        # ckpt = torch.load(self.config.updated_base_model_path, map_location=torch.device('cpu'))
        ckpt = torch.load(self.config.updated_base_model_path)
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.load_state_dict(ckpt)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.params_learning_rate, momentum=0.9)

        # Move the model to the GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Statically set, got from data preparation component print
        data_size = 0

        for idx_epoch in tqdm(range(self.config.params_epochs)):

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    data_size = 372
                else:
                    model.eval()
                    data_size = 93

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in train_loader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / data_size
                epoch_acc = running_corrects.double() / data_size
                # print(f'\n {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        self.save_model(self.config.trained_model_path, model.state_dict())
        print("Training complete!")

    def main(self):
        os.makedirs(self.config.root_dir, exist_ok=True)
        dataloader = DataPreparation(config=self.config)
        train_data = dataloader.load_data()
        self.train(train_data)

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)
