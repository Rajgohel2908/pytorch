import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

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

total  = 0
correct = 0

with torch.no_grad():
    for i, (image,label) in enumerate(data_loader):
                output = model(image)
                _, predicted = torch.max(output,1)

                correct += (predicted == label).sum().item()
                total += label.size(0)

print(f"Total:{total}\nCorrect:{correct}\n\nAccuracy:{correct / total * 100}")