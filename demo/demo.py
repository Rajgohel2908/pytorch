import torch
import torch.nn as nn  # nn module mein saare layers hote hain

# 1. Input Data (Tera flattened tensor)
# Shape: (64 images, 784 pixels)
input_tensor = torch.rand(64, 784)

# 2. Define the Layer
# in_features: 784 (Jitna data aa raha hai - MATCH HONA ZAROORI HAI)
# out_features: 10 (Jitne neurons humein chahiye output mein)
layer = nn.Linear(in_features=784, out_features=10)

# 3. Pass data through the layer (Forward Pass)
output = layer(input_tensor)

print("Input Shape:", input_tensor.shape)
print("Output Shape:", output.shape)