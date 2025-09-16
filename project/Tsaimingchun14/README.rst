HMM-Train: A Numerical Software Package for Hidden Markov Model Training
========================================================================

Basic Information
=================

**Repository URL:** https://github.com/Tsaimingchun14/hmm-train

**About field (GitHub):**
A numerical software package for training Hidden Markov Models using the
Baum-Welch algorithm, with Python interface and C++ core for performance.

Problem to Solve
================

Hidden Markov Models (HMMs) are widely used in sequence modeling problems,
including speech recognition, finance (market regime detection), and robotics
(terrain or interaction state estimation).

The key challenge in applying HMMs is **parameter estimation** when hidden
states are not observed. This is solved by the **Baum-Welch algorithm**, a
special case of Expectation-Maximization (EM).

Mathematically, given an observation sequence :math:`O_{1:T}`, Baum-Welch
iteratively maximizes the log-likelihood:

.. math::

   \log P(O_{1:T} \mid A, B, \pi)

with respect to the transition matrix :math:`A`, emission matrix :math:`B`, and
initial state distribution :math:`\pi`.

While Python packages like ``hmmlearn`` provide these functions, they are not
optimized for **large-scale numerical problems** and are harder to extend into
numerical computing coursework (C++ integration, testing, performance
evaluation).

Our project aims to provide a **hybrid Python-C++ implementation** of
Baum-Welch, designed for:

* Numerical stability (scaling in Forward-Backward).
* Performance (C++ core for heavy computation).
* Extensibility (allow future addition of Forward-Backward decoding, Viterbi,
  and continuous emissions).

Prospective Users
=================

* **Students and researchers** who need a lightweight, educational HMM toolkit
  with clear code and tests.
* **Numerical computing practitioners** who care about performance and
  correctness of EM-based algorithms.
* **Robotics/finance demo users** who want to model hidden state processes from
  time series.

Users will:

1. Provide observation sequences.
2. Train an HMM with Baum-Welch.
3. Query log-likelihoods, learned parameters, or decode hidden states with
   Forward-Backward/Viterbi.

System Architecture
===================

**Workflow**

1. Input: Discrete observation sequences.
2. Training: Baum-Welch algorithm implemented in C++ (efficient matrix
   multiplications, scaling for underflow).
3. Python Interface: Pybind11 or Cython wrapper for Python usability.
4. Output:

   * Learned transition matrix :math:`A`, emission matrix :math:`B`,
     initial distribution :math:`\pi`.
   * Log-likelihood at each iteration.
   * (Optional) Decoded state sequences using Viterbi / posterior probabilities
     using Forward-Backward.

**Constraints**

* Initial focus on **discrete observation HMMs**.
* Number of hidden states must be specified by user.
* Future extensions possible: continuous emissions (Gaussian), online
  Baum-Welch.

**Modularization**

* ``hmm/core/`` (C++ implementation: forward, backward, Baum-Welch).
* ``hmm/python/`` (Python bindings, API interface).
* ``hmm/tests/`` (unit + integration tests).
* ``examples/`` (toy datasets: weather, finance).

API Description
===============

Python user script example:

.. code-block:: python

   from hmmtrain import HMM

   # initialize model
   model = HMM(n_states=3, n_obs=4, max_iter=50)

   # fit model on observation sequence
   obs_seq = [0, 1, 2, 3, 1, 0, 2, 1]
   model.fit(obs_seq)

   # get results
   print("Log-likelihood:", model.log_likelihoods_)
   print("Transition matrix:", model.A_)
   print("Emission matrix:", model.B_)

Engineering Infrastructure
==========================

* **Build system:**
  * C++: CMake
  * Python bindings: Pybind11

* **Version control:** Git + GitHub (feature branches, pull requests).

* **Testing:**
  * Python: ``pytest``
  * C++: ``Catch2`` or ``GoogleTest``
  * Continuous Integration: GitHub Actions (optional).

* **Documentation:**
  * Sphinx (Python API docs).
  * Doxygen (C++ core).

* **Examples:**
  * Synthetic recovery test.
  * Weather toy dataset.
  * Stock/robot sensor demo (if time permits).

Schedule
========

Development timeline (8 weeks):

* Week 1 (10/20): Repository setup, project skeleton (C++ + Python binding structure).
* Week 2 (10/27): Implement forward algorithm (C++), unit test with toy HMM.
* Week 3 (11/03): Implement backward algorithm (C++), unit test with log-likelihood calculation.
* Week 4 (11/10): Implement Baum-Welch training loop (C++), add scaling for underflow. Add basic convergence tests.
* Week 5 (11/17): Python binding with Pybind11, expose training and parameter retrieval.
* Week 6 (11/24): Expand testing framework (synthetic recovery, monotonic likelihood). Regression tests.
* Week 7 (12/01): Add demo examples (weather, finance). Documentation draft.
* Week 8 (12/08): Final polishing: functional tests, prepare presentation/demo.

References
==========

1. Rabiner, L. R. (1989). *A tutorial on Hidden Markov Models and selected
   applications in speech recognition.* Proceedings of the IEEE.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*
3. hmmlearn: https://github.com/hmmlearn/hmmlearn
4. Pybind11 documentation: https://pybind11.readthedocs.io
