import torch

# 3 Images, 5 Classes (0-4)
# Yeh raw output hai jo model ne diya
logits = torch.tensor([
    [0.1, 0.2, 0.9, 0.1, 0.0],  # Image 1
    [5.0, 1.0, 0.2, 0.3, 0.5],  # Image 2
    [0.1, 0.1, 0.1, 0.8, 0.1]   # Image 3
])

# Task 1: Argmax use karke winners (indices) nikal
predictions = logits.argmax(dim=-1)

print(predictions)