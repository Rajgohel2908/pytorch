import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),            
    nn.Linear(128, 10)   
)

dummy_input = torch.rand(64, 784)

prediction = model(dummy_input)

print("Prediction Shape:", prediction.shape)
