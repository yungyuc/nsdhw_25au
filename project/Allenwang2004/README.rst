
========================================
Consensus ADMM
========================================


Basic Information
===================

* A high-performance Consensus ADMM optimization toolkit designed for
    solving distributed convex problems in large-scale machine learning and
    numerical optimization tasks.
* GitHub Repository URL:
    https://github.com/Allenwang2004/consensus-admm
* The library provides:
        * Low-level C++ core with fast linear algebra routines, parallel updates,
            and memory-efficient structures.



Problem to Solve & Prospective Users
======================================

Because of the increasing size of datasets and models, many optimization
problems in machine learning and statistics are naturally distributed across
multiple nodes. Consensus ADMM is a powerful algorithm to solve such problems
efficiently by decomposing them into smaller subproblems that can be solved in
parallel while enforcing consensus among the variables.

These problems typically appear in:
        • Distributed Lasso / ElasticNet regression: split data across machines,
            reach shared model.
        • Federated Learning: multiple clients optimize local models while
            maintaining global consistency.
        • PCA / Matrix Completion: decompose large matrices with consensus
            constraints.


System Architecture
=====================

Input
        • Objective functions f_i(x_i) encoded as callable Python functions.
        • Optional proximal operators for non-smooth terms (e.g., L1, indicator).
        • Consensus constraints (e.g., x_i = z) implicitly handled.
Output
        • Converged primal variables x_i, consensus estimate z, dual variables u_i,
            and convergence history.
        • Logs / profiling / residuals for debugging and diagnostics.

Constraints
        • Problem must be convex and separable (i.e., sum f_i(x_i) + consensus
            constraints).
        • Supports smooth + non-smooth objectives.
        • Current version supports synchronous updates (async under consideration).



API Description
=================

Python Interface

.. code:: python

    from consensus_admm import ConsensusADMM

    def local_loss_i(x): ...
    def proximal_g_i(x, rho): ...

    solver = ConsensusADMM(
        num_agents=10,
        local_losses=[local_loss_1, ..., local_loss_10],
        local_prox_ops=[prox_g1, ..., prox_g10],
        rho=1.0,
        max_iters=1000,
        tol=1e-4
    )

    results = solver.solve()
    z = results['consensus']
    x_list = results['locals']
    history = results['residuals']

    Main Class

    class ConsensusADMM:
        def __init__(
            self,
            num_agents: int,
            local_losses: list[Callable[[np.ndarray], float]],
            local_prox_ops: list[Callable[[np.ndarray, float], np.ndarray]],
            rho: float = 1.0,
            max_iters: int = 1000,
            tol: float = 1e-4
        ):
            """Initialize ADMM solver with agent-wise loss and proximal
            updates."""

        def solve(self) -> dict:
            """Run ADMM iterations and return consensus + local solutions."""



Engineering Infrastructure
============================

1. Build System
    • CMake for compiling the C++ backend
    • pybind11 for C++ ↔ Python bindings
    • Optional: Support for CUDA backend in future versions

2. Licensing
    • Apache 2.0 License

3. Testing Framework
    • C++ Unit Tests:
    • Numerical correctness of updates
    • Python Tests using pytest:
    • API compliance and regression tests
    • Python-level functional examples (Lasso, Ridge)

4. Documentation
    • README.md: install, usage, FAQ
    • Python: rich docstrings + examples


Schedule
==========

Week Milestone

* 09/27  Repository setup, literature review, project skeleton(C++ core + Python binding)
* 10/04  Implement primal/dual updates and support for L1, L2 proximal operators
* 10/11  Benchmark on synthetic distributed Lasso && Ridge regression and python binding with Pybind11
* 10/18  Add plotting / convergence diagnostics and prepare for PCA / Matrix Completion knowledge
* 10/25  Extend to PCA / Matrix Completion via ADMM formulation
* 11/01  Optimize performance, memory usage; profile bottlenecks
* 11/08  Finalize README, write docs
* 11/15  Stretch goal: multi-threading

References
============

1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2010).
    Distributed Optimization and Statistical Learning via the Alternating
    Direction Method of Multipliers
2. pybind11 Documentation: https://pybind11.readthedocs.io/
3. CMake Documentation: https://cmake.org/documentation/
4. Eigen C++ Linear Algebra: https://eigen.tuxfamily.org