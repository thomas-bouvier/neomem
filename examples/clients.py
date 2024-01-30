import torch
import neomem
import random
import ctypes
import numpy as np

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __getitem__(self, index):
        tensor = torch.full((3, 224, 224), index % K, dtype=torch.float32)
        label = index % K
        return tensor, label

    def __len__(self):
        return 5000 # Total number of samples in the dataset

# num_classes
K = 100
# rehearsal_size
N = 65
# num_candidates
C = 20
# num_representatives
R = 20
# batch_size
B = 128

aug_samples = torch.zeros(B + R, 3, 224, 224)
aug_labels = torch.randint(high=K, size=(B + R,))
aug_weights = torch.zeros(B + R)

aug_samples2 = torch.zeros(B + R, 3, 224, 224)
aug_labels2 = torch.randint(high=K, size=(B + R,))
aug_weights2 = torch.zeros(B + R)

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = MyDataset()
    loader = DataLoader(dataset=dataset, batch_size=B,
                            shuffle=True, num_workers=4, pin_memory=True)

    engine1 = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
    dsl1 = neomem.DistributedStreamLoader.create(
        engine1,
        neomem.Classification, K, N, R, C,
        ctypes.c_int64(torch.random.initial_seed()).value, 1, [3, 224, 224], neomem.CPUBuffer, False, True
    )

    engine2 = neomem.EngineLoader("tcp://127.0.0.1:1235", 1, False)
    dsl2 = neomem.DistributedStreamLoader.create(
        engine2,
        neomem.Classification, K, N, R, C,
        ctypes.c_int64(torch.random.initial_seed()).value, 1, [3, 224, 224], neomem.CPUBuffer, False, True
    )

    dsl1.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})
    dsl1.enable_augmentation(True)
    dsl1.start()

    dsl2.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})
    dsl2.enable_augmentation(True)
    dsl2.start()

    for epoch in range(4):
        for i, (inputs, target) in enumerate(loader):
            #inputs, target = inputs.cuda(), target.cuda()
            print(f"================================{i} (epoch: {epoch})")
            dsl1.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            dsl2.accumulate(inputs, target, aug_samples2, aug_labels2, aug_weights2)
            print("before wait")
            size1 = dsl1.wait()
            size2 = dsl2.wait()

            for j in range(B, size1):
                assert(aug_labels[j] == aug_samples[j, 0, 0, 0])

    dsl1.finalize()
    dsl2.finalize()
    engine1.wait_for_finalize()
    engine2.wait_for_finalize()
