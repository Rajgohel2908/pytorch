import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/mnist_cnn_v1.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("Model Loaded...")

def predict_image(image_path):
    image = Image.open(image_path)

    image = image.convert('L')

    image = ImageOps.invert(image)

    image = image.resize((28,28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    img_tensor = transform(image)

    img_tensor = img_tensor.view(1, 1, 28, 28)

    with torch.no_grad():
        output = model(img_tensor)

        probabilities = F.softmax(output, dim=1)

        confidence, predicted_class = torch.max(probabilities, 1)
        
        print(f"Predicted: {predicted_class.item()}, confidence: {confidence.item() * 100:.2f}")

image_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "my_digit.png")
predict_image(image_path)