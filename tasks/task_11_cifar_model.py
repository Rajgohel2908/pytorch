import torch
import os
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
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

loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for i, (image, label) in enumerate(data_loader):
        output = model(image)
        loss = loss_fc(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch [{epoch+1}/{epochs}] : step [{i+1}/{len(data_loader)}] : error [{loss.item():.4f}]")

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(script_dir), 'models')
os.makedirs(models_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(models_dir, 'cifer10_model.pth'))
print("Model Saved!")