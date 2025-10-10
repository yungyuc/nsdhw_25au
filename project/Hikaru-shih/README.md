Fourier-Transform-Library
================

Topic
-----
A lightweight library for Fourier Transform on small-scale signals.

Repository
----------
`Fourier-Transform-Library <https://github.com/Hikaru-shih/Fourier-Transform-Library>`_

Problem to Solve
-----------------------------
Fourier Transform is a fundamental tool in signal processing, image analysis, 
and many scientific applications. While powerful libraries such as NumPy and FFTW 
already exist, they are either too heavy for small-scale educational purposes 
or lack simple, minimal examples that can be directly reused in teaching or small projects.  

The problem to solve is:  
**“There is no lightweight and easy-to-use library that demonstrates Fourier Transform 
for small-scale signals with simple APIs and automated testing.”**

Goals
--------------------
1. **Basic Goals**
   - Implement 1D Discrete Fourier Transform (DFT).
   - Implement 1D Inverse DFT.
   - Provide simple APIs for input and output of signal arrays.
   - Validate correctness with unit tests.

2. **Extended Goals (if time allows)**
   - Implement 2D Fourier Transform (for images).
   - Compare performance with NumPy FFT.
   - Add visualization utilities for frequency spectra.
   - Provide real-world usage examples (e.g., simple signal filtering).

Technical Approach
------------------
- **Programming Language**: Python (for easier math prototyping).  
- **Testing**: pytest for unit tests.  
- **CI/CD**: GitHub Actions for continuous integration and automated testing.  
- **Documentation**: Usage examples with plots (Matplotlib).  
- **Version Control**: GitHub Pull Request workflow.  

Schedule
--------
- Week 1 (10/13): Survey existing FFT libraries and confirm project scope.  
- Week 1 (10/13): Implement 1D DFT and Inverse DFT.  
- Week 2-3 (10/26): Write unit tests and set up CI/CD pipeline.  
- Week 2-3 (10/26): Extend to 2D Fourier Transform and visualization utilities.  
- Week 4 (11/02): Compare performance with NumPy FFT and optimize code.  
- Week 5 (11/9): Finalize documentation and usage examples.  
- Week 6 (11/16): Prepare final report and presentation.  

References
----------
- Cooley–Tukey FFT Algorithm: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm  
- NumPy FFT Module: https://numpy.org/doc/stable/reference/routines.fft.html  
- FFTW Library: http://www.fftw.org  
- GitHub Actions Documentation: https://docs.github.com/en/actions