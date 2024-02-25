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
        self.skipTest("Multiple providers needed")

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
            1, [3, 224, 224], 0, neomem.CPUBuffer, False, True
        )

        engine2 = neomem.EngineLoader("tcp://127.0.0.1:1235", 1, False)
        dsl2 = neomem.DistributedStreamLoader(
            engine2,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], 0, [], neomem.CPUBuffer, False, True
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

    def test_neomem_tcp(self):
        """Test that a single rehearsal buffer can be created using the
        tcp protocol.
        """
        self.skipTest("skip")

        # num_classes
        K = 100
        # rehearsal_size
        N = 65
        # num_candidates
        C = 20
        # num_representatives
        R = 20

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], 0, [], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_buffer(self):
        """Test that a single rehearsal buffer (in standard mode)
        returns the correct representatives and doesn't cause any crash.

        The standard mode prepares a new augmented minibatch of size B + R
        at every iteration i.e., copies the last batch of size B into the new one
        and samples additional R representatives via `accumulate`.

        Parameter `size` will have a max value of B + R.
        """
        self.skipTest("skip") # validated

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
        fake = torch.tensor([1])

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Rehearsal, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 1, [3, 224, 224], 0, 0, [], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                dsl.accumulate([inputs], target, [], [aug_samples], aug_labels, aug_weights, [], fake)
                size, _ = dsl.wait()

                # Training the DNN, aug_samples should be a new instance
                # at every iteration.
                # batch = aug_samples[:size]
                # model(batch)

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
        self.skipTest("skip") # validated

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

        buf_samples = torch.zeros(R, 3, 224, 224)
        buf_labels = torch.randint(high=K, size=(R,))
        buf_weights = torch.zeros(R)
        fake = torch.tensor([1])

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Rehearsal, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 1, [3, 224, 224], 0, 0, [], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables([buf_samples], buf_labels, buf_weights, [], fake)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                dsl.accumulate([inputs], target, [])
                size, _ = dsl.wait()

                # Training the DNN
                # batch = torch.cat((inputs, buf_samples[:size]))
                # model(batch)

                for j in range(size):
                    assert torch.all(buf_samples[j] == buf_labels[j])

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_ptycho_buffer(self):
        """Test the case where a representative is composed of multiple
        samples.
        """
        self.skipTest("Fails at shutdown") # validated

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
        fake = torch.tensor([1])

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Rehearsal, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 3, [1, 256, 256], 0, 0, [], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                dsl.accumulate(
                    [inputs, amp, ph],
                    target,
                    [],
                    [aug_samples_recon, aug_amp, aug_ph],
                    aug_targets_recon,
                    aug_weights_recon,
                    [],
                    fake
                )
                size, _ = dsl.wait()

                for j in range(B, size):
                    assert torch.all(aug_samples_recon[j] == aug_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                    assert torch.all(aug_samples_recon[j] == aug_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_ptycho_buffer(self):
        """Test the case where a representative is composed of multiple
        samples.
        """
        self.skipTest("skip") # validated

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

        buf_samples_recon = torch.zeros(R, 1, 256, 256)
        buf_amp = torch.zeros(R, 1, 256, 256)
        buf_ph = torch.zeros(R, 1, 256, 256)
        buf_targets_recon = torch.zeros(R)
        buf_weights_recon = torch.zeros(R)
        fake = torch.tensor([1])

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Rehearsal, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 3, [1, 256, 256], 0, 0, [], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(
            [buf_samples_recon, buf_amp, buf_ph],
            buf_targets_recon,
            buf_weights_recon,
            [],
            fake
        )
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                dsl.accumulate(
                    [inputs, amp, ph],
                    target,
                    []
                )
                size, _ = dsl.wait()

                for j in range(size):
                    assert torch.all(buf_samples_recon[j] == buf_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                    assert torch.all(buf_samples_recon[j] == buf_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_der_buffer(self):
        self.skipTest("skip") # validated

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
        # num_representatives used for distillation
        R_distillation = 16

        buf_activations = torch.zeros(R_distillation, K)
        buf_activations_rep = torch.zeros(R_distillation, 3, 224, 224)
        fake = torch.tensor([1])

        last_inputs = None
        last_targets = None
        activations = None

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.KD, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 1, [3, 224, 224], R_distillation, 1, [K], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs],
                        last_targets,
                        [activations],
                        [],
                        fake,
                        fake,
                        [buf_activations],
                        buf_activations_rep
                    )
                    _, size = dsl.wait()

                # Training the DNN
                #
                # loss = model(inputs)
                #
                # kd_output = model(buf_activations_rep[:size])
                # loss += alpha * (mse(kd_output, buf_activations))

                activations = torch.full((B, K), 42, dtype=torch.float32)
                for i in range(target.size()[0]):
                    activations[i, :] = target[i]

                if last_inputs is not None:
                    for j in range(size):
                        assert torch.allclose(buf_activations[j], buf_activations[j, 0])
                        assert torch.allclose(buf_activations_rep[j], buf_activations_rep[j, 0, 0, 0])
                        assert buf_activations[j, 0] == buf_activations_rep[j, 0, 0, 0]

                last_inputs = inputs
                last_targets = target

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_der_buffer(self):
        self.skipTest("skip") # validated

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
        # num_representatives used for distillation
        R_distillation = R

        buf_activations = torch.zeros(R_distillation, K)
        buf_activations_rep = torch.zeros(R_distillation, 3, 224, 224)
        fake = torch.tensor([1])

        last_inputs = None
        last_targets = None
        activations = None

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.KD, K, N, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            R, 1, [3, 224, 224], R_distillation, 1, [K], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(
            [],
            fake,
            fake,
            [buf_activations],
            buf_activations_rep
        )
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs],
                        last_targets,
                        [activations],
                    )
                    _, size = dsl.wait()

                # Training the DNN
                #
                # loss = model(inputs)
                #
                # kd_output = model(buf_activations_rep)
                # loss += alpha * mse(kd_output, buf_activations)

                activations = torch.full((B, K), 42, dtype=torch.float32)
                for i in range(target.size()[0]):
                    activations[i, :] = target[i]

                if last_inputs is not None:
                    for j in range(size):
                        assert torch.allclose(buf_activations[j], buf_activations[j, 0])
                        assert torch.allclose(buf_activations_rep[j], buf_activations_rep[j, 0, 0, 0])
                        assert buf_activations[j, 0] == buf_activations_rep[j, 0, 0, 0]

                last_inputs = inputs
                last_targets = target

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_derpp_buffer(self):
        """Test that a single rehearsal buffer (in standard mode)
        returns the correct representatives and doesn't cause any crash.

        The standard mode prepares a new augmented minibatch of size B + R
        at every iteration i.e., copies the last batch of size B into the new one
        and samples additional R representatives via `accumulate`.

        Parameter `size` will have a max value of B + R.
        
        We send activations to the buffer too, useful for knowledge distillation.
        We do leverage rehearsal here, thus there is a need to augment mini-batches.
        """
        self.skipTest("skip")

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
        # num_representatives used for distillation
        R_distillation = R

        aug_samples = torch.zeros(B + R, 3, 224, 224)
        aug_labels = torch.randint(high=K, size=(B + R,))
        aug_weights = torch.zeros(B + R)
        buf_activations = torch.zeros(R_distillation, K)
        buf_activations_rep = torch.zeros(R_distillation, 3, 224, 224)

        last_inputs = None
        last_targets = None
        activations = None

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], 1, [K], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs],
                        last_targets,
                        [activations],
                        [aug_samples],
                        aug_labels,
                        aug_weights,
                        [buf_activations],
                        buf_activations_rep
                    )
                    size = dsl.wait()

                # Training the DNN
                #
                # loss = model(aug_samples[:size])
                #
                # kd_output = model(buf_activations_rep[:size])
                # loss += alpha * (mse(kd_output, buf_activations))
                activations = torch.full((B, K), 42, dtype=torch.float32)
                for i in range(len(target)):
                    activations[i] = inputs[i]

                if last_inputs is not None:
                    for j in range(B, size):
                        pass

                last_inputs = inputs
                last_targets = target

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_derpp_buffer(self):
        self.skipTest("skip")

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
        # num_representatives used for distillation
        R_distillation = R

        buf_samples = torch.zeros(R, 3, 224, 224)
        buf_labels = torch.randint(high=K, size=(R,))
        buf_weights = torch.zeros(R)
        buf_activations = torch.zeros(R_distillation, K)
        buf_activations_rep = torch.zeros(R_distillation, 3, 224, 224)

        last_inputs = None
        last_targets = None
        activations = None

        dataset = MyDataset(K)
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            1, [3, 224, 224], 1, [K], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(
            [buf_samples],
            buf_labels,
            buf_weights,
            [buf_activations],
            buf_activations_rep
        )
        dsl.start()

        for _ in range(2):
            for inputs, target in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs],
                        last_targets,
                        [activations],
                    )
                    size = dsl.wait()

                # Training the DNN
                #
                # batch = torch.cat((inputs, buf_samples[:size]))
                # loss = model(inputs)
                #
                # kd_output = model(buf_activations_rep)
                # loss += alpha * mse(kd_output, buf_activations)

                activations = torch.full((B, K), 42, dtype=torch.float32)
                for i in range(len(target)):
                    activations[i] = inputs[i]

                if last_inputs is not None:
                    for j in range(size):
                        pass

                last_inputs = inputs
                last_targets = target

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_standard_derpp_ptycho_buffer(self):
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
        # num_representatives used for distillation
        R_distillation = R

        aug_samples = torch.zeros(B + R, 1, 256, 256)
        aug_targets = torch.zeros(B + R)
        aug_weights = torch.zeros(B + R)
        aug_amp = torch.zeros(B + R, 1, 256, 256)
        aug_ph = torch.zeros(B + R, 1, 256, 256)
        buf_amp_activations = torch.zeros(R_distillation, 1, 256, 256)
        buf_ph_activations = torch.zeros(R_distillation, 1, 256, 256)
        buf_activations_rep = torch.zeros(R_distillation, 1, 256, 256)

        last_inputs = None
        last_targets = None
        last_amps = None
        last_phs = None
        activations_amp = None
        activations_ph = None

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            3, [1, 256, 256], 2, [1, 256, 256], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs, last_amps, last_phs],
                        last_targets,
                        [activations_amp, activations_ph],
                        [aug_samples, aug_amp, aug_ph],
                        aug_targets,
                        aug_weights,
                        [buf_amp_activations, buf_ph_activations],
                        buf_activations_rep
                    )
                    size = dsl.wait()

                # Training the DNN
                #
                # amp_output, ph_output = model(aug_samples[:size])
                # loss1 = criterion(amp_output, aug_amp)
                # loss2 = criterion(ph_output, aug_ph)
                # loss = loss1 + loss2
                #
                # amp_kd_output, ph_kd_output = model(buf_activations_rep[:size])
                # kd_loss1 = mse(amp_kd_output, buf_amp_activations[:size])
                # kd_loss2 = mse(ph_kd_output, buf_ph_activations[:size])
                # loss += alpha * (kd_loss1 + kd_loss2)

                activations_amp = torch.full((B, 1, 256, 256), 42, dtype=torch.float32)
                activations_ph = torch.full((B, 1, 256, 256), 42, dtype=torch.float32)
                for i in range(len(target)):
                    activations_amp[i] = inputs[i]
                    activations_ph[i] = inputs[i]

                if last_inputs is not None:
                    for j in range(B, size):
                        assert torch.all(aug_samples[j] == aug_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                        assert torch.all(aug_samples[j] == aug_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))

                last_inputs = inputs
                last_targets = target
                last_amps = amp
                last_phs = ph

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_flyweight_derpp_ptycho_buffer(self):
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
        # num_representatives used for distillation
        R_distillation = R

        buf_samples = torch.zeros(R, 1, 256, 256)
        buf_targets = torch.zeros(R)
        buf_weights = torch.zeros(R)
        buf_amp = torch.zeros(R, 1, 256, 256)
        buf_ph = torch.zeros(R, 1, 256, 256)
        buf_amp_activations = torch.zeros(R_distillation, 1, 256, 256)
        buf_ph_activations = torch.zeros(R_distillation, 1, 256, 256)
        buf_activations_rep = torch.zeros(R_distillation, 1, 256, 256)

        last_inputs = None
        last_targets = None
        last_amps = None
        last_phs = None
        activations_amp = None
        activations_ph = None

        dataset = PtychoDataset()
        loader = DataLoader(dataset=dataset, batch_size=B, shuffle=True)

        engine = neomem.EngineLoader("tcp://127.0.0.1:1234", 0, False)
        dsl = neomem.DistributedStreamLoader.create(
            engine,
            neomem.Classification, K, N, R, C,
            ctypes.c_int64(torch.random.initial_seed()).value,
            3, [1, 256, 256], 2, [1, 256, 256], neomem.CPUBuffer, False, self.verbose
        )
        dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
        dsl.enable_augmentation(True)
        dsl.use_these_allocated_variables(
            [buf_samples, buf_amp, buf_ph],
            buf_targets,
            buf_weights,
            [buf_amp_activations, buf_ph_activations],
            buf_activations_rep
        )
        dsl.start()

        for _ in range(2):
            for inputs, target, amp, ph in loader:
                if last_inputs is not None:
                    dsl.accumulate(
                        [last_inputs, last_amps, last_phs],
                        last_targets,
                        [activations_amp, activations_ph]
                    )
                    size = dsl.wait()

                # Training the DNN
                #
                # loss = model(inputs)
                #
                # amp_output, ph_output = model(buf_samples[:size])
                # loss1 = criterion(amp_output, buf_amp)
                # loss2 = criterion(ph_output, buf_ph)
                # loss += beta * (loss1 + loss2)
                #
                # amp_kd_output, ph_kd_output = model(buf_activations_rep[:size])
                # kd_loss1 = mse(amp_kd_output, buf_amp_activations[:size])
                # kd_loss2 = mse(ph_kd_output, buf_ph_activations[:size])
                # loss += alpha * (kd_loss1 + kd_loss2)

                activations_amp = torch.full((B, 1, 256, 256), 42, dtype=torch.float32)
                activations_ph = torch.full((B, 1, 256, 256), 42, dtype=torch.float32)
                for i in range(len(target)):
                    activations_amp[i] = inputs[i]
                    activations_ph[i] = inputs[i]

                if last_inputs is not None:
                    for j in range(size):
                        assert torch.all(buf_samples[j] == buf_amp[j] - torch.full((1, 256, 256), 1000, dtype=torch.float32))
                        assert torch.all(buf_samples[j] == buf_ph[j] - torch.full((1, 256, 256), 2000, dtype=torch.float32))
                        assert torch.all(buf_samples[j] == buf_ph_activations[j])

                last_inputs = inputs
                last_targets = target
                last_amps = amp
                last_phs = ph

        dsl.finalize()
        engine.wait_for_finalize()

    def test_neomem_engine_shutdown(self):
        """Test that a single rehearsal buffer can be properly shut down,
        and that a second can be started without causing any crash.
        """
        self.skipTest("Skipped")
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
                1, [3, 224, 224], 0, [], neomem.CPUBuffer, False, self.verbose
            )
            dsl.register_endpoints({"tcp://127.0.0.1:1234": 0})
            dsl.enable_augmentation(True)
            dsl.start()

            for inputs, target in loader:
                dsl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
                dsl.wait()

            dsl.finalize()
            engine.wait_for_finalize()
