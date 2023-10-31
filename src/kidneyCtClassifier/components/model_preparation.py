import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

from kidneyCtClassifier.entity.config_entity import PrepareBaseModelConfig


class ModelPreparation:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        os.makedirs(self.config.root_dir, exist_ok=True)
        self.save_model(self.config.base_model_path, model.state_dict())

        return model

    def full_model_updated(self):
        os.makedirs(self.config.root_dir, exist_ok=True)
        model = self.get_base_model()
        for name, param in model.named_parameters():
            if "fc" in name:
                model.requires_grad_(True)
            else:
                model.requires_grad_(False)

        self.save_model(self.config.updated_base_model_path, model.state_dict())

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)
