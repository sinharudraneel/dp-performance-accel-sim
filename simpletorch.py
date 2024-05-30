import torch
import torch.nn as nn
import torch.optim as optim

# Define the single-layer neural network
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate input data and target data
X = torch.randn(10, 3).to(device)
y = torch.randn(10, 1).to(device)

# Set input size and output size
input_size = 3
output_size = 1

# Initialize the model, criterion, and optimizer
model = SingleLayerNN(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model with a new sample
with torch.no_grad():
    test_sample = torch.randn(1, 3).to(device)
    prediction = model(test_sample)
    print(f'Prediction for test sample: {prediction.item():.4f}')

