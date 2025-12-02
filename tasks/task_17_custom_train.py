import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root='./my_dataset/train', transform=train_transform)
val_data = datasets.ImageFolder(root='./my_dataset/val', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
vak_loader = DataLoader(val_data, batch_size=4, shuffle=False)

class_name = train_data.classes
print(f"classes name:{class_name}")

model = models.resnet18(weights='DEFAULT')
num_fuction = model.fc.in_features
model.fc = nn.Linear(num_fuction, len(class_name))

loss_fc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        output = model(inputs)
        loss = loss_fc(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] loss [{running_loss/len(train_loader):.4f}]")

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(script_dir), 'models')
os.makedirs(models_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(models_dir, 'custom_train.pth'))