import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

input_data = torch.rand(64,784)
target_data = torch.randint(0, 10,(64,))

for epoch in range(100):
    prediction = model(input_data)
    loss = loss_fn(prediction, target_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch : {epoch} & Loss : {loss}")