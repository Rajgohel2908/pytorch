import torch

data = torch.rand(64,28,28)
print(data.shape)

flat_tensor = data.view(64,-1)
print(flat_tensor.shape)