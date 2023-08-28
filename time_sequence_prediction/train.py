from __future__ import print_function
import os
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import struct
import threading
import time
matplotlib.use('Agg')


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


def update_pid():
    # This function will run in a separate thread
    file_path = 'benchmarking/pids/pytorch.pid'

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


if __name__ == '__main__':
    # Create a thread for updating the PID
    pid_thread = threading.Thread(target=update_pid)
    pid_thread.start()

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('benchmarking/temp/pytorch/traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(opt.steps):
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            y = pred.detach().numpy()

    # Wait for the PID thread to finish
    pid_thread.join()
