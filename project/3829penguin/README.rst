JacobiPCG: Jacobi Preconditioned Conjugate Gradient Method
=========================================================

Basic Information
=================

**Repository:** `https://github.com/3829penguin/JacobiPCG <https://github.com/3829penguin/JacobiPCG>`_

Problem Statement
=================

In advanced process nodes and 2.5D/3D integrated packages, the rapid increase in 
power density gives rise to severe local hotspots and intricate inter-layer heat 
transfer phenomena. These effects must be captured using high-resolution 
finite-difference (FDM) models to ensure thermal accuracy.

However, FDM generates extremely large grid systems, leading to very large sparse 
linear equations. Classical iterative solvers such as Conjugate Gradient (CG) 
suffer from long runtimes and slow convergence, especially on ill-conditioned 
systems.

To address this challenge, we implement the **Jacobi Preconditioned Conjugate 
Gradient (JacobiPCG)** method, which applies Jacobi preconditioning to CG, 
significantly accelerating convergence and reducing computational cost.

Target Users
============

Chip and system designers who require accurate and efficient thermal verification 
during both early design exploration and sign-off analysis.They need to solve large sparse 
linear systems efficiently to ensure thermal reliability in advanced integrated
packages.JacobiPCG provides a robust and efficient solution for these users.

System Overview
===============

I aim to solve sparse linear systems with matrix sizes in the range of 10^5*10^5 to 10^7*10^7 , corresponding to several hundred thousand to several million unknowns.

1. **Input:** Sparse matrix :math:`A` in CSR format, right-hand side vector :math:`b`.
2. **Output:** Approximate solution vector :math:`x`.
3. **Constraints:** The matrix :math:`A` must be an **SPD-matrix** (Symmetric Positive Definite).

API Description
===============

**Function Signature:**

.. code-block:: c++

    vector<double>& JacobiPCG(CSR A, vector<double>& b);

**Example Usage:**

.. code-block:: c++

    #include <vector>
    #include <iostream>
    #include <JacobiPCG.h>

    int main() {
        // Parse CSR matrix A and vector b
        CSR A;
        vector<double> b;
        vector<double> x;
        x = JacobiPCG(A, b);
        return 0;
    }

Engineering Infrastructure
==========================

* **Build System:** CMake
* **Version Control:** GitHub
* **Testing Framework:** Google Test (C++)
* **Documentation:** README.rst

Development Schedule
====================

* **Week 1 (10/6):** Implement CSR matrix parser.
* **Week 2 (10/13):** Implement Jacobi preconditioner (diagonal extraction & inverse).
* **Week 3 (10/20):** Implement Conjugate Gradient solver.
* **Week 4 (10/27):** Integrate preconditioner with CG (JacobiPCG solver).
* **Week 5 (11/03):** Add parallel acceleration (OpenMP/MKL), or maybe see if we can make some optimizations based on this method.
* **Week 6 (11/10):** Finalize API interface.
* **Week 7 (11/17):** Validate solver accuracy and performance.
* **Week 8 (11/24):** Conduct functional tests, prepare presentation/demo, and finalize documentation.

References
==========

1. Saad, Yousef. *Iterative Methods for Sparse Linear Systems.* SIAM, 2003.
