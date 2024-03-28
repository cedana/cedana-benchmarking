import torch

# Check if CUDA is available
torch.cuda.init()
device=torch.device("cuda:0")
# Create two tensors
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)

# Perform addition
c = a + b

print(c)

# If we used CUDA, let's move the result back to CPU to print it (not strictly necessary if just printing)
if device.type == 'cuda':
    print(c.to('cpu'))

