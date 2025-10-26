from setuptools import setup, Extension
import pybind11
import os
import sys
import subprocess

def find_blas_library():
    """Try to find available BLAS library"""
    # Check for MKL headers first (most important check)
    mkl_header_paths = [
        '/usr/include/mkl_cblas.h',
        '/opt/intel/mkl/include/mkl_cblas.h', 
        '/usr/local/include/mkl_cblas.h'
    ]
    
    mkl_headers_found = any(os.path.exists(path) for path in mkl_header_paths)
    
    if mkl_headers_found:
        # Also check if MKL libraries exist
        mkl_lib_paths = [
            '/usr/lib/x86_64-linux-gnu',
            '/opt/intel/mkl/lib/intel64',
            '/usr/local/lib'
        ]
        
        for path in mkl_lib_paths:
            if os.path.exists(os.path.join(path, 'libmkl_core.so')):
                return 'mkl'
    
    # Check for OpenBLAS
    openblas_paths = [
        '/usr/lib/x86_64-linux-gnu/libopenblas.so',
        '/usr/lib/libopenblas.so',
        '/usr/local/lib/libopenblas.so'
    ]
    
    if any(os.path.exists(path) for path in openblas_paths):
        return 'openblas'
    
    # Fallback to generic BLAS
    return 'blas'

def get_compile_args():
    """Get compilation arguments based on system and available libraries"""
    args = ['-std=c++17', '-O3', '-DNDEBUG']
    
    # Avoid problematic flags that may not work on all systems
    if sys.platform != 'darwin':  # Not macOS
        args.append('-march=native')
    
    return args

def get_link_args_and_libs():
    """Get linking arguments and libraries based on detected BLAS"""
    blas_type = find_blas_library()
    
    if blas_type == 'mkl':
        # MKL configuration
        return {
            'libraries': ['mkl_intel_lp64', 'mkl_sequential', 'mkl_core', 'm', 'dl'],
            'library_dirs': ['/usr/lib/x86_64-linux-gnu', '/opt/intel/mkl/lib/intel64'],
            'define_macros': [('USE_MKL', None)]
        }
    elif blas_type == 'openblas':
        # OpenBLAS configuration
        return {
            'libraries': ['openblas', 'm'],
            'library_dirs': [],
            'define_macros': []
        }
    else:
        # Generic BLAS configuration (fallback)
        return {
            'libraries': ['blas', 'm'],
            'library_dirs': [],
            'define_macros': []
        }

# Get BLAS configuration
blas_config = get_link_args_and_libs()

module = Extension('_matrix', 
                   sources=['matrix.cpp'],
                   include_dirs=[pybind11.get_include()],
                   extra_compile_args=get_compile_args(),
                   libraries=blas_config['libraries'],
                   library_dirs=blas_config['library_dirs'],
                   define_macros=blas_config['define_macros'])

setup(name='matrix',
      version='1.0',
      description='Matrix multiplication module',
      ext_modules=[module])