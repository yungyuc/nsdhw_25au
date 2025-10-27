FFTCore
=======

Basic Information
================

   Repository: `FFTCore <https://github.com/Hikaru-shih/FFTCore>`_

   Project Description: FFTCore is a lightweight Fourier Transform library focused on optimizing performance and numerical stability for small-scale signals.
   It aims to explore efficient algorithmic designs for DFT and FFT, balancing speed, accuracy, and implementation simplicity.

   System Requirement: 
   * Python 3.10+ with pip  
   * numpy>=1.24
   * pytest>=7.0
   * (optional) numba>=0.59 for performance tests
   * Works on Linux, macOS, and Windows environments.  
   * No GPU or external FFT library required.

Problem to Solve
================

   Existing FFT frameworks prioritize scalability and multi-threaded performance but are not optimized for small or medium-sized data arrays.
   This leads to inefficiency and limited flexibility when experimenting with algorithmic variants.

   FFTCore addresses this gap by providing a minimal yet optimized FFT implementation that focuses on precision, simplicity, and benchmarking for small-scale signals.

Prospective Users
=================

   * Students and researchers who need a lightweight and transparent FFT toolkit for studying the Discrete Fourier Transform and testing optimization ideas.
   * Numerical computing practitioners who care about precision, efficiency, and reproducibility in small-scale signal processing experiments.
   * Embedded system or hardware developers who require minimal FFT implementations for low-memory or real-time environments.

   Users will:

      1. Provide small 1D or 2D signal arrays as input data.
      2. Perform DFT and inverse DFT using FFTCore APIs.
      3. Compare accuracy and runtime against NumPy FFT or other standard libraries.

System Architecture
===================

   Workflow
      1. Input: 1D or 2D discrete signal arrays (small-scale data, e.g. 16–1024 samples).
      2. Processing: FFT and Inverse FFT implemented in C++ (optimized butterfly operations and memory layout).
      3. Python Interface: Pybind11 wrapper for user-level access and numerical experiments.
      4. Output:
         * Complex frequency-domain representation (magnitude and phase).
         * Reconstructed signal via Inverse FFT.
         * Optional: Benchmark reports and visualization of performance comparison (NumPy FFT vs. FFTCore).

   Constraints
      1. Focus on small-scale discrete Fourier Transform (DFT/FFT) computations.
      2. Designed for real-valued and complex-valued inputs.
      3. Limited optimization for GPU or large-scale distributed workloads (to be considered for future versions).
      4. Numerical precision aligned with IEEE 754 double-precision floating-point standard.

   Modularization
      1. fftcore/cpp/ — C++ implementation of DFT/FFT and inverse transform algorithms.
      2. fftcore/python/ — Python bindings using Pybind11 (API and utility functions).
      3. fftcore/tests/ — Unit tests for numerical correctness and benchmark validation.
      4. examples/ — Demonstrations: simple sine wave, square wave, and image frequency visualization.

API Description
================

Python user script example
.. code-block:: python

   from fftcore import FFT
   import numpy as np

   # initialize model
   fft = FFT(n=32)

   # prepare signal
   t = np.linspace(0, 1, 32, endpoint=False)
   x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

   # perform forward FFT
   X = fft.forward(x)

   # perform inverse FFT
   x_rec = fft.inverse(X)

   # get results
   print("Original signal:", x)
   print("Frequency domain:", X)
   print("Reconstructed signal:", x_rec)

Engineering Infrastructure
==========================

   * Build system: C++: CMake (standalone executable build, debug/release modes)
   * Version control: Git + GitHub (feature branches, pull requests)
   * Testing: C++: GoogleTest (numerical validation, benchmark comparison)
   * Documentation: Doxygen (C++ core), README with usage and performance notes
   * Examples: FFT benchmark test, frequency visualization, runtime scaling


Schedule
========

   Development timeline (5 weeks):

   * Week 1 (10/24-31): Project initialization. Set up repository, CMake build, and base directory structure. Review DFT and FFT algorithm design.
   * Week 2 (11/01-07): Implement 1D DFT and inverse DFT. Verify numerical correctness on small signals.
   * Week 3 (11/08-14): Implement radix-2 FFT and IFFT. Add benchmark tool for comparing DFT vs FFT performance.
   * Week 4 (11/15-21): Optimize computation (e.g., loop unrolling, memory layout). Add runtime measurement and accuracy test cases.
   * Week 5 (11/22-30): Finalize documentation and Doxygen comments. Generate benchmark report and complete final submission.


References
==========
   1. https://hackmd.io/@8dSak6oVTweMeAe9fXWCPA/H1y3L57Yd
   2. Duan, B., Wang, W., Li, X., Zhang, C., Zhang, P., & Sun, N. (2011, December). Floating-point mixed-radix FFT core generation for FPGA and comparison with GPU and CPU. In 2011 International Conference on Field-Programmable Technology (pp. 1-6). IEEE.
   3. https://ithelp.ithome.com.tw/m/articles/10322646