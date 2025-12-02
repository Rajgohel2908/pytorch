import torch 
import torch.nn as nn

vocab_size = 10
embedding_size = 5
hidden_dim = 10

embed = nn.Embedding(vocab_size,embedding_size)
rnn = nn.RNN(embedding_size,hidden_dim,batch_first=True)

input = torch.tensor([[2, 3, 9, 1, 6]])
print(f"input shape:{input.shape}")

vector = embed(input)
print(f"embedding shape : {vector.shape}")

output, hidden = rnn(vector)

print(f"RNN output shape : {output.shape}")
print(f"Final hidden state shape : {hidden.shape}")