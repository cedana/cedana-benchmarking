from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F
import os
import struct
import threading
import time

def update_pid():
    # This function will run in a separate thread
    file_path = 'benchmarking/pids/pytorch-regression.pid'

    while True:
        # Get the process ID
        pid = os.getpid()
        # Convert the process ID to a 32-bit integer
        pid_int32 = pid & 0xFFFFFFFF  # Ensure it fits within 32 bits
        # Open the file in binary write mode
        with open(file_path, 'wb') as file:
            # Write the int32 data to the file using struct.pack
            file.write(struct.pack('I', pid_int32))
        time.sleep(1)  # Wait for one second

def main():
    # Create a thread for updating the PID
    pid_thread = threading.Thread(target=update_pid)
    pid_thread.start()

    POLY_DEGREE = 500
    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5

    # Rest of your code for model training
    fc = torch.nn.Linear(W_target.size(0), 1)
    # ...

    for batch_idx in count(1):
        # Get data
        batch_x, batch_y = get_batch()

        # Reset gradients
        fc.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(fc(batch_x), batch_y)
        loss = output.item()

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-0.1 * param.grad)

        # Stop criterion
        if loss < 1e-3:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))

    # Wait for the PID thread to finish
    pid_thread.join()

if __name__ == "__main__":
    main()
