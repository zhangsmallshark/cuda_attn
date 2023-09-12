from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_ROOT_DIR="/soft/compilers/cudatoolkit/cuda-11.8.0"

setup(
    name='linear_cutlass',
    include_dirs=[f"{CUDA_ROOT_DIR}/include", "/home/czh5/seq/cudnn_attn/cutlass/include", "/home/czh5/seq/cudnn_attn/cutlass/tools/util/include", "/home/czh5/seq/cudnn_attn/cutlass/examples/common"],
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