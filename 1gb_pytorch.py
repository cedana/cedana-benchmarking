import torch
import torch.nn as nn
import time

# Define the neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10000, 5000)
        self.fc2 = nn.Linear(5000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 500)
        self.fc5 = nn.Linear(500, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 10)
        self.fc8 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# Initialize the model
model = MyModel()

# Create a large input tensor
input_tensor = torch.randn(1000, 10000)

# Set the duration for the long-running process (e.g., 1 hour)
run_duration = 3600  # seconds

start_time = time.time()

while time.time() - start_time < run_duration:
    # Process the tensor through the model
    output = model(input_tensor)

    # Add a sleep delay to control the loop's execution speed
    time.sleep(1)  # sleep for 1 second

    # Optionally, print output or status to verify the process
    print("Processed at:", time.ctime())

print("Long-running process completed.")