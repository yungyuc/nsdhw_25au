================
VLSIGR
================

Basic Information
=================

**GitHub Repository**: https://github.com/Duck997/VLSIGR

**Project Description**: A high-performance VLSI global router
implementation that incorporates adaptive algorithms and intelligent
scoring mechanisms to achieve competitive routing quality in integrated
circuit design.

**System Requirements**:
- **C++**: C++14 or later with OpenMP support
- **Python**: Python 3.7+ with Cython for C++ binding
- **Dependencies**: NumPy, Matplotlib (optional for visualization)

Problem to Solve
================

VLSI (Very Large Scale Integration) global routing is a critical step
in the physical design of integrated circuits, responsible for
determining the routing paths for millions of interconnects while
satisfying design constraints such as capacity limits, wirelength
minimization, and timing requirements.

**Field and Industry Context**
The semiconductor industry faces increasing challenges as modern ICs
continue to scale with billions of transistors. The computational
complexity of routing algorithms has become a significant bottleneck in
the design flow, directly impacting time-to-market and design quality.

**Physics and Mathematics**
The routing problem involves:
- **Graph Theory**: Representing the routing grid as a graph with capacity constraints
- **Optimization Theory**: Minimizing wirelength while satisfying capacity constraints
- **Dynamic Programming**: Implementing efficient path-finding algorithms
- **Adaptive Algorithms**: Dynamic parameter adjustment based on routing context


Prospective Users
=================

**Target Users and Applications**
- **VLSI Design Engineers**: Professionals working on large-scale
  integrated circuit design, integrating the router into existing EDA
  toolchains
- **EDA Tool Developers**: Engineers developing electronic design
  automation software, using the router for performance benchmarking
  and algorithm comparison
- **Research Institutions**: Academic researchers studying routing
  algorithms and optimization, utilizing the router as a research
  platform for investigating new techniques
- **Educational Purposes**: Teaching advanced routing concepts and
  algorithm design in academic settings

**User Workflow**
Users will interact with the router through programmatic interfaces
(C++ and Python APIs), providing input data in standard formats (e.g.,
ISPD benchmarks) and receiving optimized routing solutions with
performance metrics. The router can be integrated into existing EDA
toolchains or used as a standalone library for research and development
purposes.

System Architecture
===================

**Input Processing**
- **File Parsing**: Reading routing benchmark files in multiple formats:
  - **ISPD Benchmarks**: Standard global routing contest format (.gr files)
  - **Custom Formats**: Extensible parser for user-defined input
    specifications
  - **Legacy Support**: Compatibility with existing EDA tool formats
  - **Future Extensions**: Framework for additional benchmark formats as
    needed
- **Grid Graph Construction**: Creating efficient 2D grid representations
  with capacity and congestion tracking
- **Net Decomposition**: Converting multi-pin nets into two-pin connections
  using MST algorithms
- **Priority Sorting**: Ordering nets and connections based on routing
  difficulty and congestion metrics

**Core Routing Pipeline**
1. **Pattern Routing Phase**
   - **L-shape Routing**: Initial routing using L-shaped patterns
   - **Z-shape Routing**: Alternative routing using Z-shaped patterns
   - **Monotonic Routing**: Constrained routing following monotonic
     constraints
   - Capacity constraint validation and overflow detection

2. **Advanced Scoring System**
   - **Net-level Score Function**: Multi-factor evaluation considering
     overflow, cost, and network complexity
     - Prioritizes difficult networks for early processing
     - Balances overflow resolution with wirelength optimization
   
   - **Two-pin Score Function**: Adaptive scoring with multiple modes for different routing phases
   
   - **Edge Cost Calculation**: Dynamic cost evaluation with history-based
     penalty
     - Incorporates historical overflow data for congestion awareness
     - Adaptive penalty weights based on routing phase and global
       congestion
     - Encourages selection of less congested paths

3. **HUM (Hybrid Unilateral Monotonic) Optimization**
   - **Iterative Improvement**: Dynamic programming-based path optimization
   - **Adaptive Expansion Strategy**: Dual approach with uniform and
     directional expansion
   - **Progressive Refinement**: Dynamic adjustment based on routing progress

4. **Multi-stage Routing Flow**
   - **Initial Routing**: Pattern-based routing with L-shape, Z-shape, and
     Monotonic strategies using early exploration cost functions
   - **HUM Optimization**: Advanced routing with adaptive expansion and
     history-based cost using balanced optimization strategies
   - **Overflow Resolution**: Iterative rip-up and rerouting with dynamic
     strategy switching based on routing phase
   - **Convergence**: Progressive refinement with adaptive cost function
     selection until routing quality targets are met

5. **Layer Assignment** *(Third-party Module, see [3]_ for details)*
   - Multi-layer routing optimization using provided LayerAssignment
     module
   - Via minimization and wirelength optimization

**Output Generation**
- **Routing Results**: Detailed routing paths for all nets
- **Performance Metrics**: Execution time, overflow statistics, wirelength
  measurements
- **Visualization Tools**: Graphical analysis capabilities
  - **Traffic Visualization**: Color-coded congestion maps showing routing
    density
  - **Layer Analysis**: Multi-layer routing visualization with via
    representation
  - **Overflow Detection**: Visual identification of capacity violations



API Description
===============

**C++ Interface**

.. code-block:: cpp

   #include "vlsigr.hpp"  // Main router interface with ISPD parser
   // and visualization
   
   // Load ISPD benchmark file
   ISPDParser::ispdData* data = ISPDParser::parse_file("adaptec1.gr");
   
   // Initialize router with ISPD benchmark data
   VLSIGR::GlobalRouting router;
   router.init(*data);
   
   // Configure optimization parameters
   router.setMode(VLSIGR::Mode::BALANCED);  // Use balanced mode
   router.enableAdaptiveScoring(true);
   router.enableHUMOptimization(true);
   
   // Execute routing
   router.route();
   
   // Access results
   auto results = router.getResults();
   auto metrics = router.getPerformanceMetrics();
   
   // Generate visualization
   VLSIGR::Visualization viz;
   viz.generateMap(data, results, "routing_result.ppm");
   
   // Clean up resources
   router.cleanup();
   
   
**Python Interface**

.. code-block:: python

   import vlsigr
   from vlsigr import Mode
   
   # Create router instance
   router = vlsigr.GlobalRouter()
   
   # Load ISPD benchmark file
   router.load_ispd_benchmark("adaptec1.gr")
   
   # Configure parameters
   router.set_mode(Mode.BALANCED)  # Use balanced mode
   router.enable_adaptive_scoring(True)
   router.enable_hum_optimization(True)
   
   # Execute routing
   results = router.route()
   
   # Access metrics
   metrics = router.get_metrics()
   print(f"Execution time: {metrics.execution_time}s")
   print(f"Total overflow: {metrics.total_overflow}")
   print(f"Wirelength: {metrics.wirelength}")
   
   # Generate visualization
   router.visualize_results(results, "routing_result.ppm")
   
   # Clean up resources
   router.cleanup()

Engineering Infrastructure
==========================

1. Automatic Build System - Makefile

   The project uses a unified Makefile-based build system for both C++
   and Python interfaces with cross-platform compatibility and static
   linking for server deployment.

2. License - Academic Research Use

   This project is developed for academic research purposes. The core
   routing algorithms and adaptive scoring mechanisms are original
   implementations by the project author. Third-party components
   (LayerAssignment Module) are used under their respective academic
   research licenses.

3. Testing Framework

   - **Contest Benchmark Testing**: Validate routing quality and performance
     using ISPD contest benchmarks.
   - **Algorithm Verification**: Test HUM optimization and adaptive scoring
     strategies.
   - **Cross-platform Compatibility**: Ensure compatibility across different
     environments.

4. Documentation

   - **User Documentation**: `README.md` (project overview, features, license
     information, and references).
   - **Code Documentation (We put function docs in codebase! while
     high-level docs in README.md)**:
     - C++: Inline comments for core algorithms (e.g., `// HUM optimization
       with adaptive expansion`).
     - Python: Docstrings for API functions (e.g., `def route(): """Execute
       routing with current configuration..."""`).
   - **Algorithm Documentation**: Detailed explanations of HUM optimization,
     scoring functions, and routing strategies in code comments.

Schedule
========

8-week development plan from 10/13 to 12/7, 2025.

The timeline begins with foundation work including ISPD benchmark analysis
and core routing algorithm implementation, followed by advanced scoring
systems and HUM optimization development. The middle phase focuses on
performance optimization and comprehensive ISPD benchmark validation, while
the final weeks concentrate on API development, documentation, and
visualization tools.

This structured approach ensures thorough algorithm development and testing
before moving to user interface and delivery components, with built-in
flexibility for the complex integration and validation phases that are
critical for VLSI routing systems.

*   **Week 1 (10/13 - 10/19): Foundation & Initial Algorithm**
    *   Project setup, ISPD benchmark analysis, initial algorithm design
    *   Core routing algorithm implementation, basic data structures
*   **Week 2 (10/20 - 10/26): Environment & Scoring System**
    *   Development environment setup, dependency installation, build system
      configuration
    *   Advanced scoring system development, multi-mode algorithm
      implementation
*   **Week 3 (10/27 - 11/02): HUM & Adaptive Strategies**
    *   HUM optimization strategy development, adaptive routing strategy
      design
    *   Algorithm refinement and initial testing
*   **Week 4 (11/03 - 11/09): Performance & Platform Optimization**
    *   Performance analysis and bottleneck identification, initial
      performance optimization
    *   Cross-platform compatibility testing, server deployment optimization
*   **Week 5 (11/10 - 11/16): Benchmark Validation & Integration**
    *   ISPD benchmark validation, comprehensive performance analysis
    *   System preliminary integration, core module functionality testing
*   **Week 6 (11/17 - 11/23): Full System Integration & Debugging**
    *   Complete system integration, bug fixes and stability testing
    *   Edge case and robustness testing
*   **Week 7 (11/24 - 11/30): API & Documentation**
    *   C++ and Python API development and refinement
    *   User documentation writing, code comments and high-level documentation
*   **Week 8 (12/01 - 12/07): Visualization & Finalization**
    *   Visualization tools development and integration
    *   Final testing and validation, project completion and presentation preparation

References
==========

1. ISPD 2008 Global Routing Contest. https://www.ispd.cc/contests/08/ispd08rc.html
2. W.-H. Liu, Y.-L. Li, and C.-K. Koh. "A fast maze-free routing congestion
   estimator with hybrid unilateral monotonic routing." 2012 IEEE/ACM
   International Conference on Computer-Aided Design (ICCAD), San Jose, CA,
   USA, 2012, pp. 713-719.
3. W.-H. Liu, W.-C. Kao, Y.-L. Li, and K.-Y. Chao. "NCTU-GR 2.0: Multithreaded
   Collision-Aware Global Routing With Bounded-Length Maze Routing." IEEE
   Transactions on Computer-Aided Design of Integrated Circuits and Systems,
   vol. 32, no. 5, pp. 709-722, May 2013. DOI: 10.1109/TCAD.2012.2235124
4. VLSI Physical Design Automation: Theory and Practice. Sait, Sadiq M., and
   Youssef, Habib.
5. Global Routing in VLSI Design. Cong, Jason, and Shinnerl, Joseph R.


