import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from torchvision import models


class PredictionPipeline:

    def __init__(self, image_path):
        self.image_path = image_path

    def predict(self):
        try:
            # load the model for prediction
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
            model.load_state_dict(torch.load('model/model.pth'))
            model.eval()

            # Load and preprocess the unseen image
            image = Image.open(self.image_path)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

            with torch.no_grad():
                output = model(input_batch)

            # Get the predicted class
            _, predicted_class = output.max(1)

            # Map the predicted class to the class name
            class_names = ['normal', 'tumor']  # Make sure these class names match your training data
            predicted_class_name = class_names[predicted_class.item()]

            return [{"image": predicted_class_name}]

        except Exception as e:
            return [{"INFO": "No kidney ct image !"}]
