import asyncio
import grpc
import task_pb2_grpc
import task_pb2
import time
import torch
import torch.nn as nn

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

async def main():
    # Initialize the model
    model = MyModel()

    # Create a large input tensor
    input_tensor = torch.randn(1000, 10000)

    # Set the duration for the long-running process (e.g., 1 hour)
    run_iters = 20
    
    for i in range(run_iters):
        # Process the tensor through the model
        output = model(input_tensor)
        time.sleep(1)
        if i % 1000 == 0:
            # Every 1000 iters, print output to verify the process
            print("Processed at:", time.ctime())

    print("Process completed.")

    # Take checkpoint once processing completes
    channel = grpc.aio.insecure_channel('localhost:8080')
    import sys
    jobID = sys.argv[1]
    print("jobID = ", jobID)
    dump_args = task_pb2.DumpArgs()
    dump_args.Dir = "/terminal-ckpt/"
    dump_args.Type = task_pb2.DumpArgs.LOCAL
    dump_args.JobID = jobID
    stub = task_pb2_grpc.TaskServiceStub(channel)
    dump_resp = await stub.Dump(dump_args)
    print(dump_resp)

if __name__ == "__main__":
    asyncio.run(main())
