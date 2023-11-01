import os
from pathlib import Path
from urllib.parse import urlparse
import mlflow
import torch
import torch.nn as nn
from numpy import double
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from kidneyCtClassifier.entity.config_entity import EvaluationConfig
from kidneyCtClassifier.utils.common import transform_data, save_json


class MlflowModelEvaluation:
    def __init__(self, config=EvaluationConfig):
        self.config = config
        self.transform = transform_data()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.score = {}

        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model = model.load_state_dict(torch.load(self.config.path_of_model))

        # os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/andcoul/kidney-ct-classification-pytorch.mlflow'
        # os.environ['MLFLOW_TRACKING_USERNAME'] = 'andcoul'
        # os.environ['MLFLOW_TRACKING_PASSWORD'] = '3c2d382f23858d4d2a20e9d5f187d6bdef1e83c2'

    def valid_generator(self):
        dataset = ImageFolder(str(self.config.training_data), transform=self.transform['val'])
        datasets = self.train_val_dataset(dataset)
        # print(len(datasets['train']))
        # print(len(datasets['val']))
        # The original dataset is available in the Subset class
        # print(datasets['train'].dataset)
        image_list = DataLoader(dataset=datasets['val'], batch_size=self.config.params_batch_size, pin_memory=True,
                                shuffle=False)

        return image_list

    def load_model(self):
        # Load the saved model
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
        model.load_state_dict(torch.load(self.config.path_of_model))
        model.eval()

        # Create a new model with the correct final layer
        new_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

        # Copy the weights and biases from the loaded model to the new model
        new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
        new_model.fc.bias.data = model.fc.bias.data[0:2]

        return model

    def inference(self):
        # Perform inference
        model = self.load_model()
        running_loss = 0.0
        running_corrects = 0

        criterion = nn.CrossEntropyLoss()

        for inputs, labels in tqdm(self.valid_generator()):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / 93
            epoch_acc = running_corrects.double() / 93
            self.score['loss'] = double(epoch_loss)
            self.score['Acc'] = double(epoch_acc)
        self.save_score()

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        # mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score['loss'], "accuracy": self.score['Acc']}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.load_model(), "model", registered_model_name="KidneyResNet18Model")
            else:
                mlflow.pytorch.log_model(self.load_model(), "model")

    @staticmethod
    def train_val_dataset(dataset, val_split=0.20):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

    def save_score(self):
        scores = {"loss": str(self.score['loss']), "accuracy": str(self.score['Acc'])}
        save_json(path=Path("scores.json"), data=scores)
