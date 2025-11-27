import torch

# Tensor A: Shape (2, 3) -> 2 rows, 3 columns
A = torch.rand(2, 3)

# Tensor B: Shape (2, 3) -> 2 rows, 3 columns
B = torch.rand(2, 3).T

# Hum matrix multiplication karne ki koshish kar rahe hain
result = A @ B
print(result)