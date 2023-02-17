# Neomem

C++ data loader with rehearsal for Torch. Based on PyBind11 and [Mochi](https://www.mcs.anl.gov/research/projects/mochi/).

## Requirements

- Python
- PyTorch
- CUDA
- MPI
- Thallium

Verbs requires MOFED to support CUDA. More specifically, it requires the kernel "peer memory" API which is only available in MOFED's version of IB drivers.
