import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

print("Data Loaded!!!")
print(f"Total Batches: {len(train_data)}")