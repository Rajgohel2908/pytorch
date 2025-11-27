import torch
import os
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),            
    nn.Linear(128, 10)   
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 3

for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images = images.view(-1,28*28)

        prediction = model(images)
        loss = loss_fn(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    
print("\nExam time! Testing on unseen data... 📝")

test_data = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model.eval() 

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")

if accuracy > 90:
    print("Oye hoye! Topper nikla tera model! 🎓")
elif accuracy > 80:
    print("Pass ho gaya, par distinction nahi aayi. 👍")
else:
    print("Fail! Wapas padhai karwao isko. 🤦‍♂️")

script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_location)
models_folder = os.path.join(project_root, 'models')
os.makedirs(models_folder, exist_ok=True)
save_path = os.path.join(models_folder, "mnist_v1.pth")
torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path} successfully! 💾")