import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)

        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output
model = CNN()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/cifer10_model.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("model loaded")

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

        probability = F.softmax(output, 1)
        conf, pred = torch.max(probability, 1)

        ans = classes[pred.item()]
        conf_score = conf.item() * 100

    print(f"prediction:{ans}\ncofidence:{conf_score:.2f}")

image_name = "image.jpg" 
image_path = os.path.join(script_dir, image_name)

predict_image(image_path)