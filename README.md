Simple skeleton of C++ data loader with rehearsal for Torch.
Based on PyBind11.

undefined symbol: THPVariableClass -> TORCH_USE_RTLD_GLOBAL=YES
or better https://github.com/pytorch/pytorch/issues/38122#issuecomment-629470997

spack env create thallium spack.yaml
spack env activate thallium
spack concretize
spack install

. /home/tbouvier/Dev/spack/share/spack/setup-env.sh && spack env activate thallium
make clean && make && make test

python torch_test.py
