import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os

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

model = generator()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/gan_generator.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

noise = torch.randn(16, 100)

with torch.no_grad():
    fake_img = model(noise)

    fig, axes = plt.subplots(4, 4, figsize=(8,8))
    