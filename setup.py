from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_ROOT_DIR="/usr/local/cuda"

setup(
    name='linear_cutlass',
    include_dirs=[f"{CUDA_ROOT_DIR}/include", "/home/guanhuawang/cutlass/include", "/home/guanhuawang/cutlass/tools/util/include", "/home/guanhuawang/cutlass/examples/common"],
    ext_modules=[
        CUDAExtension('linear_cutlass', [
            'linear_cutlass.cpp',
            'linear_cutlass_kernel.cu',
        ],
        extra_compile_args={'nvcc': ['-std=c++17']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
