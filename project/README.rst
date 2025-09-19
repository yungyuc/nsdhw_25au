===============================
AgAMGPCG: Aggregation-based Algebraic Multigrid Preconditioned Conjugate Gradient Method
===============================

Basic Information
========

**Repository:** https://github.com/3829penguin/AgAMG
  
Problem to Solve 
==========
In advanced process nodes and 2.5D/3D stacked packaging, 
the high power density leads to significant local hotspots and complex inter-layer heat transfer behavior, 
which must be captured using high-resolution FDM models.
However, FDM generates a large number of grids, 
resulting in extremely large linear systems. Traditional solvers such as CG require prohibitively long runtimes, 
and CG often exhibits poor convergence on large-scale systems.
Therefore, we employ aggregation-based AMG as a preconditioner for CG to improve convergence speed.

Prospective Users
============

Chip and system designers, who require accurate and efficient thermal verification during both early design and sign-off stages.

System Architecture
===================
1.Input: Sparse matrix in CSR format :math:`A`, right-hand side vector :math:`b`, threshold :math:`n`.
2.Output: Approximate solution vector :math:`x`.
3.Constraints: The matrix :math:`A` should be M-matrix.
M-matrix is a matrix whose off-diagonal entries are less than or equal to zero and whose eigenvalues have nonnegative real parts. 

API Description
===============

Function:
.. code-block:: c++
    vector<double>& AgAMG(CSR A, vector<double>& b);
Example Usage:
.. code-block:: c++
    #include <vector>
    #include <iostream>
    #include <AgAMGPCG.h>
    int main() {
        //Parsing CSR and b
        CSR A;
        vector<double> b;
        vector<double> x;
        x =  AgAMG(A, b);
        return 0;
    }
Engineering Infrastructure
==========================
1. Build system: CMake
2. Version control: GitHub
3. Testing framework: C++: Google Test
4. Documentation: README.rst

Schedule
========
* Week 1 (10/20): Implement Parser to read sparse matrix in CSR format from file.
* Week 2 (10/27): Implement Aggregate algorithm.
* Week 3 (11/03): Implement AMG algorithm.
* Week 4 (11/10): Implement CG algorithm.
* Week 5 (11/17): Parallel acceleration.
* Week 6 (11/24): Complete API interface.
* Week 7 (12/01): Final Validation of code.
* Week 8 (12/08): Complete the final polishing by conducting functional tests, preparing the presentation and demo, and compiling the references.
==========
1. Notay, Yvan. "An aggregation-based algebraic multigrid method." Electron. Trans. Numer. Anal 37.6 (2010): 123-146.