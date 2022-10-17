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

aug_samples = torch.zeros(B + R, 3, 224, 224)
aug_labels = torch.randint(high=K, size=(B + R,))
aug_weights = torch.zeros(B + R)

aug_samples2 = torch.zeros(B + R, 3, 224, 224)
aug_labels2 = torch.randint(high=K, size=(B + R,))
aug_weights2 = torch.zeros(B + R)

N_recon = 1000

aug_samples_recon = torch.zeros(B + R, 1, 128, 128)
aug_targets_recon = torch.zeros(B + R, 1, 128, 128)
aug_weights_recon = torch.zeros(B + R)

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # https://github.com/pytorch/pytorch/issues/5059
    values = np.random.rand(5000, 3, 224, 224)
    labels = np.random.randint(0, K, 5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=B,
                        shuffle=True, num_workers=4, pin_memory=True)

    dsl1 = rehearsal.DistributedStreamLoader(
        rehearsal.Classification, K, N, C, ctypes.c_int64(torch.random.initial_seed()).value, 0, "tcp://127.0.0.1:1234", 1, [3, 224, 224], False)
    dsl1.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})
    dsl2 = rehearsal.DistributedStreamLoader(
        rehearsal.Classification, K, N, C, ctypes.c_int64(torch.random.initial_seed()).value, 1, "tcp://127.0.0.1:1235", 1, [3, 224, 224], False)
    dsl2.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})

    for epoch in range(4):
        for i, (inputs, target) in enumerate(loader):
            #inputs, target = inputs.cuda(), target.cuda()
            print(f"================================{i} (epoch: {epoch})")
            dsl1.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            dsl2.accumulate(inputs, target, aug_samples2, aug_labels2, aug_weights2)
            print("before wait")
            dsl1.wait()
            dsl2.wait()


    values_recon = np.random.rand(5000, 1, 128, 128)
    target_recon = np.random.rand(5000, 1, 128, 128)
    dataset = MyDataset(values_recon, target_recon)
    loader = DataLoader(dataset=dataset, batch_size=B,
                        shuffle=True, num_workers=4, pin_memory=True)

    dsl_recon = rehearsal.DistributedStreamLoader(
        rehearsal.Reconstruction, 1, N_recon, C, ctypes.c_int64(torch.random.initial_seed()).value, 0, "tcp://127.0.0.1:1234", 2, [1, 128, 128], False)

    for epoch in range(4):
        for i, (inputs, target) in enumerate(loader):
            #inputs, target = inputs.cuda(), target.cuda()
            print(f"================================{i} (epoch: {epoch})")
            dsl_recon.accumulate(inputs, target, aug_samples_recon, aug_targets_recon, aug_weights_recon)
            size = dsl2.wait()
            print(f"Received {size - B} samples from other nodes")
