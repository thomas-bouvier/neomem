import ctypes
import os
import random
import unittest

import mpi4py
import neomem
import numpy as np
import pytest
import torch

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, K):
        self.K = K

    def __getitem__(self, index):
        tensor = torch.full((3, 224, 224), index % self.K, dtype=torch.float32)
        label = index % self.K
        return tensor, label

    def __len__(self):
        return 5000 # Total number of samples in the dataset

class PtychoDataset(Dataset):
    def __getitem__(self, index):
        tensor = torch.full((1, 256, 256), index, dtype=torch.float32)
        amp = torch.full((1, 256, 256), index + 1000, dtype=torch.float32)
        ph = torch.full((1, 256, 256), index + 2000, dtype=torch.float32)
        return tensor, 0, amp, ph

    def __len__(self):
        return 2500

def skip_or_fail_gpu_test(test, message):
    """Fails the test if GPUs are required, otherwise skips."""
    if int(os.environ.get('NEOMEM_TEST_GPU', 0)):
        test.fail(message)
    else:
        test.skipTest(message)

class TorchTests(unittest.TestCase):

    verbose = True

    def setup(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def test_gpu_required(self):
        if not torch.cuda.is_available():
            skip_or_fail_gpu_test(self, "No GPUs available")

    def test_neomem_multiple_clients(self):
        self.skipTest("Multiple providers")

        # num_classes
        K = 100
        # rehearsal_size
        N = 65
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples = torch.zeros(B + R, 3, 224, 224)
        aug_labels = torch.randint(high=K, size=(B + R,))
        aug_weights = torch.zeros(B + R)

        aug_samples2 = torch.zeros(B + R, 3, 224, 224)
        aug_labels2 = torch.randint(high=K, size=(B + R,))
        aug_weights2 = torch.zeros(B + R)

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B,
                                shuffle=True, num_workers=4, pin_memory=True)

        engine1 = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl1 = neomem.DistributedStreamLoader(
            engine1,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], neomem.CPUBuffer, False, True
        )

        engine2 = neomem.EngineLoader("tcp://127.0.0.1:1235", 1, False)
        dsl2 = neomem.DistributedStreamLoader(
            engine2,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], neomem.CPUBuffer, False, True
        )

        dsl1.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})
        dsl1.enable_augmentation(True)
        dsl1.start()

        dsl2.register_endpoints({'tcp://127.0.0.1:1234': 0, 'tcp://127.0.0.1:1235': 1})
        dsl2.enable_augmentation(True)
        dsl2.start()

        for _ in range(4):
            for inputs, target in loader:
                dsl1.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
                dsl2.accumulate(inputs, target, aug_samples2, aug_labels2, aug_weights2)
                size1 = dsl1.wait()
                dsl2.wait()

                for j in range(B, size1):
                    assert torch.all(aug_samples[j] == aug_labels[j])

    def test_neomem_standard_buffer(self):
        """Test that a single rehearsal buffer (in standard mode)
        returns the correct representatives and doesn't cause any crash.

        The standard mode prepares a new augmented minibatch of size B + R
        at every iteration i.e., copies the last batch of size B into the new one
        and samples additional R representatives via `accumulate`.

        Parameter `size` will have a max value of B + R.
        """
        # num_classes
        K = 100
        # rehearsal_size
        N = 65
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples = torch.zeros(B + R, 3, 224, 224)
        aug_labels = torch.randint(high=K, size=(B + R,))
        aug_weights = torch.zeros(B + R)

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                dsl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
                size = dsl.wait()

                for j in range(B, size):
                    assert torch.all(aug_samples[j] == aug_labels[j])

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_buffer(self):
        """Test that a single rehearsal buffer (in flyweight mode)
        returns the correct representatives and doesn't cause any crash.

        The flyweight mode returns representatives only as a minibatch
        of size R at every iteration, and this object is allocated only once
        i.e., not passed from Python at every iteration via `accumulate`.

        Users have to concatenate representatives of size R to an input
        minibatch of size R manually to obtain an augmented minibatch of
        size B + R.

        Parameter `size` will have a max value of R.
        """
        # num_classes
        K = 100
        # rehearsal_size
        N = 65
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples = torch.zeros(R, 3, 224, 224)
        aug_labels = torch.randint(high=K, size=(R,))
        aug_weights = torch.zeros(R)

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(aug_samples, aug_labels, aug_weights)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                dsl.accumulate(inputs, target)
                size = dsl.wait()

                for j in range(B, size):
                    assert torch.all(aug_samples[j] == aug_labels[j])

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_ptycho_buffer(self):
        """Test the case where a representative is composed of multiple
        samples.
        """
        self.skipTest("skip")
        # num_classes
        K = 1
        # rehearsal_size
        N = 650
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples_recon = torch.zeros(B + R, 1, 256, 256)
        aug_targets_recon = torch.zeros(B + R)
        aug_weights_recon = torch.zeros(B + R)
        aug_amp = torch.zeros(B + R, 1, 256, 256)
        aug_ph = torch.zeros(B + R, 1, 256, 256)

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            3, [1, 256, 256], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                dsl.accumulate(
                    inputs,
                    target,
                    amp,
                    ph,
                    aug_samples_recon,
                    aug_targets_recon,
                    aug_weights_recon,
                    aug_amp,
                    aug_ph
                )
                size = dsl.wait()

                for j in range(B, size):
                    assert torch.all(aug_samples_recon[j] == aug_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                    assert torch.all(aug_samples_recon[j] == aug_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_ptycho_buffer(self):
        """Test the case where a representative is composed of multiple
        samples.
        """
        # num_classes
        K = 1
        # rehearsal_size
        N = 650
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples_recon = torch.zeros(R, 1, 256, 256)
        aug_targets_recon = torch.zeros(R)
        aug_weights_recon = torch.zeros(R)
        aug_amp = torch.zeros(R, 1, 256, 256)
        aug_ph = torch.zeros(R, 1, 256, 256)

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            3, [1, 256, 256], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(aug_samples_recon, aug_targets_recon, aug_weights_recon, aug_amp, aug_ph)
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                dsl.accumulate(inputs, target, amp, ph)
                size = dsl.wait()

                for j in range(B, size):
                    assert torch.all(aug_samples_recon[j] == aug_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                    assert torch.all(aug_samples_recon[j] == aug_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_engine_shutdown(self):
        """Test that a single rehearsal buffer can be properly shut down,
        and that a second can be started without causing any crash.
        """
        # num_classes
        K = 100
        # rehearsal_size
        N = 65
        # num_candidates
        C = 20
        # num_representatives
        R = 20
        # batch_size
        B = 32

        aug_samples = torch.zeros(B + R, 3, 224, 224)
        aug_labels = torch.randint(high=K, size=(B + R,))
        aug_weights = torch.zeros(B + R)

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        for _ in range(2):
            engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
            dsl = neomem.DistributedStreamLoader.create(
                engine,
                neomem.Classification, K, N, R, C,
                ctypes.c_int64(torch.random.initial_seed()).value,
                1, [3, 224, 224], neomem.CPUBuffer, False, self.verbose
            )
            dsl.register_endpoints({'tcp://127.0.0.1:1234': 0})
            dsl.enable_augmentation(True)
            dsl.start()

            for inputs, target in loader:
                dsl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
                dsl.wait()

            dsl.finalize()
            engine.wait_for_finalize()
