import torch
import neomem
import random
import ctypes
import numpy as np

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, values, labels):
        super().__init__()
        self.values = values
        self.labels = labels

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


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

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

aug_samples = torch.zeros(B + R, 3, 224, 224, device=device)
aug_labels = torch.randint(high=K, size=(B + R,), device=device)
aug_weights = torch.zeros(B + R, device=device)

aug_samples2 = torch.zeros(R, 3, 224, 224, device=device)
aug_labels2 = torch.randint(high=K, size=(R,), device=device)
aug_weights2 = torch.zeros(R, device=device)

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # https://github.com/pytorch/pytorch/issues/5059
    values = np.random.rand(5000, 3, 224, 224)
    labels = np.random.randint(0, K, 5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

    ##### `standard` buffer
    print("========================standard")

    engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
    dsl = neomem.DistributedStreamLoader.create(
        engine,
        neomem.Classification, K, N, R, C,
        ctypes.c_int64(torch.random.initial_seed()).value, 1, [3, 224, 224], neomem.CPUBuffer, False, True
    )
    dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
    dsl.enable_augmentation(True)
    dsl.start()

    for epoch in range(4):
        for i, (inputs, target) in enumerate(loader):
            inputs, target = inputs.to(device), target.to(device)
            print(f"=========={i} (epoch: {epoch})")
            dsl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            print("before wait")
            dsl.wait()

    dsl.finalize()
    engine.wait_for_finalize()

    ##### `flyweight` buffer
    print("========================flyweight")

    engine2 = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
    dsl2 = neomem.DistributedStreamLoader.create(
        engine2,
        neomem.Classification, K, N, R, C,
        ctypes.c_int64(torch.random.initial_seed()).value, 1, [3, 224, 224], neomem.CPUBuffer, False, True
    )
    dsl2 = engine2.get_loader()
    dsl2.register_endpoints({'tcp://127.0.0.1:1234': 0})
    dsl2.enable_augmentation(True)
    dsl2.use_these_allocated_variables(aug_samples2, aug_labels2, aug_weights2)
    dsl2.start()

    for epoch in range(4):
        for i, (inputs, target) in enumerate(loader):
            inputs, target = inputs.to(device), target.to(device)
            print(f"=========={i} (epoch: {epoch})")
            dsl2.accumulate(inputs, target)
            print("before wait")
            dsl2.wait()

    dsl2.finalize()
    engine2.wait_for_finalize()
