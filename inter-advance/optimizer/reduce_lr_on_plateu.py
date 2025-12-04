import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class simpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        return self.fc(x)

model = simpleNN()
optimizers = {
    "SGD": lambda params: torch.optim.SGD(params, lr=0.01),
    "SGD_Momentum": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "Adam": lambda params: torch.optim.Adam(params, lr=0.001),
    "AdamW": lambda params: torch.optim.AdamW(params, lr=0.001, weight_decay=0.01),
    "RMSProp": lambda params: torch.optim.RMSprop(params, lr=0.001)
}
loss_fc = nn.CrossEntropyLoss()

def train_one_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau (
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )

    for batch, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fc(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step(total_loss)

    avg_loss = total_loss / len(loader)
    return avg_loss

for opt_name, opt in optimizers.items():
    model = simpleNN()
    optimizer = opt(model.parameters())

    loss = train_one_epoch(model, optimizer, data_loader)
    print(opt_name, "->", loss)