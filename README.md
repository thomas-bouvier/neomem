# Neomem

C++ data loader with rehearsal for Torch. Based on PyBind11 and [Mochi](https://www.mcs.anl.gov/research/projects/mochi/).

## Building

### Requirements

- Python
- PyTorch
- CUDA
- MPI
- libfabric built with CUDA support
- Thallium

If these dependencies are installed inside a Spack environment, don't forget to `activate` it before building Neomem.

### Compiling Neomem using CMake



## Providers

### Verbs

If using provider `verbs`, make sure IPoIB is enabled and that an interface appears as `UP` when running `ip link show`.

### RDMA+CUDA

Device registration should be enabled. To use RDMA+CUDA, your only options are providers `ofi+shm` (shared-memory provider from libfabric, which supports gdr copy) and `verbs`.

If using `verbs`: you need MOFED to support CUDA. More specifically, it requires the kernel "peer memory" API which is only available in MOFED's version of IB drivers. If running into issues with MOFED, check that the command `grep ib_register_peer_memory_client /proc/kallsyms` outputs something similar:

```console
ffffffffc09c3595 r __kstrtab_ib_register_peer_memory_client     [ib_core]
ffffffffc09c35b4 r __kstrtabns_ib_register_peer_memory_client   [ib_core]
ffffffffc09bd54c r __ksymtab_ib_register_peer_memory_client     [ib_core]
ffffffffc09b9620 T ib_register_peer_memory_client       [ib_core]
```

`ucx` is another option that also supports CUDA, though we don't think anybody tested it just yet :) the code is there though. However, `na+sm` (shared-memory plugin from mercury) is not GPU-enabled.