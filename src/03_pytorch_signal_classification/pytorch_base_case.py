import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Load the data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Print statistics every epoch
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    
# Evaluate the model on test data
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100*correct/total:.2f}%")

# In this template, there are several variables that can be changed to increase the accuracy and precision of the results:

# learning_rate: This controls the step size taken during gradient descent, and a smaller learning rate may lead to slower but more precise convergence.
# num_epochs: This controls the number of times the entire dataset is passed through the model during training. Increasing the number of epochs may allow the model to learn more complex patterns, but could also lead to overfitting.
# batch_size: This controls the number of samples that are processed together before backpropagating the error signal. Larger batch sizes may speed up training, but smaller batches can lead to more precise gradients.
# Net architecture: Changing the neural network architecture can significantly affect the accuracy and precision of the model. The number of layers, the size of each layer, and the activation functions used are all important factors to consider when designing a neural network.
