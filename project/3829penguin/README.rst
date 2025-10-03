PCG-CP: PCG Solver with Configurable Preconditioners
=========================================================

Basic Information
=================

The PCG-CP (Preconditioned Conjugate Gradient – Configurable Preconditioners) project is a C++ library designed to efficiently solve large-scale sparse linear systems that typically arise from finite-difference modeling of thermal problems in advanced IC and package designs.
It provides a unified interface for the Conjugate Gradient (CG) solver enhanced with configurable preconditioners, enabling faster convergence on ill-conditioned symmetric positive definite (SPD) systems.
**Repository:** `https://github.com/3829penguin/PCG-CP <https://github.com/3829penguin/PCG-CP>`_

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

To address this challenge, we implement the **Preconditioned Conjugate 
Gradient (PCG)** method, which applies some method of preconditioning to CG, 
significantly accelerating convergence and reducing computational cost.

Target Users
============

Chip and system designers who require accurate and efficient thermal verification 
during both early design exploration and sign-off analysis.They need to solve large sparse 
linear systems efficiently to ensure thermal reliability in advanced integrated
packages.PCG provides a robust and efficient solution for these users.

System Overview
===============

I aim to solve sparse linear systems with matrix sizes in the range of 10^5*10^5 to 10^7*10^7 , corresponding to several hundred thousand to several million unknowns.

1. **Input:** Sparse matrix :math:`A` in CSR format, right-hand side vector :math:`b`, preconditioner type, max iterations, and tolerance.
2. **Output:** Approximate solution vector :math:`x`.
3. **Constraints:** The matrix :math:`A` must be an **SPD-matrix** (Symmetric Positive Definite).

API Description
===============

**Preconditioner Options:**

.. code-block:: c++

    enum class PreconditionerType {
        Jacobi,
        GaussSeidel
        //others maybe
    };

**Unified Function Signature:**

.. code-block:: c++

    vector<double>& PCG_Solver(
        CSR A, 
        vector<double>& b, 
        PreconditionerType preconditioner = PreconditionerType::Jacobi,
        int max_iter = 1000, 
        double tol = 1e-8);

**Example Usage:**

.. code-block:: c++

    #include <vector>
    #include <iostream>
    #include <PCG_Solver.h>

    int main() {
        CSR A;
        vector<double> b, x;

        // Default: Jacobi preconditioner
        x = PCG_Solver(A, b);

        // Switch to Gauss–Seidel preconditioner
        x = PCG_Solver(A, b, PreconditionerType::GaussSeidel);

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
* **Week 1 (10/6):** Implement CSR matrix parser and verify the correctness of the three CSR arrays (row_ptr, col_idx, values) using small test matrices. Set up the unit testing framework (Google Test)for future automation.
* **Week 2 (10/13):** Implement Jacobi preconditioner (diagonal extraction & inverse) and verify its correctness automatically through predefined test cases comparing computed diagonals and inverses with analytical results.
* **Week 3 (10/20):** Implement Conjugate Gradient solver, and verify its correctness by automatically comparing solver results with those from a direct solver (e.g., LU) on test problems.
* **Week 4 (10/27):** Integrate Jacobi preconditioner with CG (JacobiPCG) and extend automated tests to include convergence rate and residual monitoring. Automatic testing will ensure PCG converges within expected iteration counts.
* **Week 5 (11/03):** Add parallel acceleration (OpenMP/MKL) for JacobiPCG. Implement performance regression tests that automatically log runtime and speedup compared to the serial version.
* **Week 6 (11/10):** Implement Gauss–Seidel preconditioner and automatically compare its convergence and runtime against JacobiPCG in the test suite.
* **Week 7 (11/17):** Extend solver to support multiple preconditioners via PreconditionerType enum. Update automatic testing to include parameterized test cases for each preconditioner type.
* **Week 8 (11/24):** Conduct full functional and performance regression tests, automatically generate test reports, and finalize documentation and presentation/demo materials.

References
==========

1. Saad, Yousef. *Iterative Methods for Sparse Linear Systems.* SIAM, 2003.
