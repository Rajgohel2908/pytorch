import torch
import json
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms,models
import requests

model = models.resnet18(weights='DEFAULT')
model.eval()

labels = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json')
labels = json.loads(labels.text)

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

        probability = F.softmax(output, 1)
        conf, pred = torch.max(probability, 1)

        ans = labels[pred.item()]
        conf_score = conf.item() * 100

    print(f"prediction:{ans}\ncofidence:{conf_score:.2f}")

image_name = "image.jpg" 
predict_image(image_name)