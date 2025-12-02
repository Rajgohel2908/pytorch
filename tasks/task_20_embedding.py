import torch
import torch.nn as nn

vocab_size = 10
embedding_dim = 5

embed = nn.Embedding(vocab_size, embedding_dim)
input = torch.tensor([2, 3, 9, 1, 6])

out = embed(input)

print(f"output : {out}\nshape : {out.shape}")