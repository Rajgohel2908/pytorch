import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)

        img = img.view(img.size(0), 1, 28, 28)
        return img

g = generator()
d = discriminator()

loss_fc = nn.BCELoss()
d_optimizer = optim.Adam(d.parameters(), lr=0.0002)
g_optimizer = optim.Adam(g.parameters(), lr=0.0002)

epochs = 50

for epoch in range(epochs):
    for i, (image,_) in enumerate(data_loader):
        batch_size = image.size(0)

        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        d_optimizer.zero_grad()

        image = image.view(batch_size, -1)
        outputs = d(image)
        d_loss_real = loss_fc(outputs, real_label)

        z = torch.randn(batch_size, 100)
        fake_img = g(z)
        outputs = d(fake_img.detach())
        d_loss_fake = loss_fc(outputs, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        outputs = d(fake_img)
        g_loss = loss_fc(outputs, real_label)
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch[{epoch}/{epochs}] : D_loss[{d_loss.item():.4f}] : G_loss[{g_loss.item():.4f}]")

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(script_dir), 'models')
os.makedirs(models_dir, exist_ok=True)
torch.save(g.state_dict(), os.path.join(models_dir, 'gan_generator.pth'))
torch.save(d.state_dict(), os.path.join(models_dir, 'gan_discriminator.pth'))
print("Model Saved!")