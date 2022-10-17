import torch
import rehearsal
import random
import ctypes
import numpy as np

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, values, labels):
        super(MyDataset, self).__init__()
        self.values = values
        self.labels = labels

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


# num_classes
K = 100
# num_representatives
N = 65
# num_candidates
C = 20
# num_samples
R = 20
# batch_size
B = 128

H = 25
W = 25

aug_samples = torch.zeros(B + R, 3, H, W)
aug_labels = torch.randint(high=K, size=(B + R,))
aug_weights = torch.zeros(B + R)

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # https://github.com/pytorch/pytorch/issues/5059
    values = np.random.rand(500, 3, H, W)
    labels = np.random.randint(0, K, 5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=B,
                        shuffle=True, num_workers=0, pin_memory=True)

    dsl = rehearsal.DistributedStreamLoader(
        rehearsal.Classification, K, N, C, ctypes.c_int64(torch.random.initial_seed()).value, 0, "tcp://127.0.0.1:1234", 1, [3, H, W], False)
    dsl.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})

    for epoch in range(400):
        for i, (inputs, target) in enumerate(loader):
            #inputs, target = inputs.cuda(), target.cuda()
            print(f"================================{i} (epoch: {epoch})")
            dsl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            print("before wait")
            size = dsl.wait()
            print(f"Received {size - B} samples from other nodes")

    print("Finished")