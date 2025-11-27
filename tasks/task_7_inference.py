import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image, ImageOps

model = nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/mnist_v1.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("Model Loaded")

def prediction_fn(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.view(1,784)

    with torch.no_grad():
        output = model(img_tensor)
        probablities = torch.nn.functional.softmax(output, dim=-1)
        confidence, predicted = torch.max(probablities, 1)

        print(f"\n🧠 Prediction: {predicted.item()}")
        print(f"📊 Confidence: {confidence.item()*100:.2f}%")

image_name = "my_digit.png"
image_path = os.path.join(script_dir, image_name)

prediction_fn(image_path)