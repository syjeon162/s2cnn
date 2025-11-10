# pylint: disable=R,C,E1101
from collections import namedtuple
import cupy as cp  # pylint: disable=E0401

CUDA_NUM_THREADS = 1024
CUDA_MAX_GRID_DIM = 2**16 - 1


def get_blocks(n, num_threads):
    n_per_instance = (n + num_threads * CUDA_MAX_GRID_DIM - 1) // (num_threads * CUDA_MAX_GRID_DIM)
    return (n + num_threads * n_per_instance - 1) // (num_threads * n_per_instance)


Stream = namedtuple('Stream', ['ptr'])

def compile_kernel(kernel, filename, functionname):
    """
    Compile a CUDA kernel using CuPy's built-in compilation.
    
    Args:
        kernel (str): CUDA kernel source code
        filename (str): Name for the kernel (for debugging/identification)
        functionname (str): Name of the kernel function
    
    Returns:
        cupy.RawKernel: Compiled kernel function
    """
    # Use CuPy's RawKernel for direct CUDA compilation
    return cp.RawKernel(kernel, functionname)