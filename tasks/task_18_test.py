from PIL import Image
import torch
import torch.nn as nn
import os
from torchvision import transforms, models

classes = ['Phone', 'Wallet']

model = models.resnet18(weights=None)

num_function = model.fc.in_features
model.fc = nn.Linear(num_function, len(classes))

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../models/custom_train.pth')

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("model loaded")

def predict(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

        print(f"prediction : {classes[pred.item()]}")
        print(f"confidence : {conf.item()*100:.2f}%")

image_name = "test_image.jpg" 
image_path = os.path.join(script_dir, image_name)

predict(image_path)