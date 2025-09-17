=================
Voronoi Diagram Toolkit
=================

Basic Information
=================

- **GitHub Repository URL**: https://github.com/ryankert01/voronoi-diagram-toolkit  
- The toolkit provides both low-level C++ APIs (for performance) and high-level Python APIs (for usability).
- **A high-performance C++/Python toolkit** for 2D Voronoi diagram construction and nearest neighbor querying, supporting computational - geometry scenarios like mesh generation. 
  Also, I'm planning to support visualization in fewer-point scenarios(fewer than 100 points).


Problem to Solve
================

Voronoi diagrams are a fundamental data structure in computational geometry, enabling spatial partitioning based on proximity. There're many applications of Voronoi diagrams described below.

Application Scenarios
  1. **Mesh Generation**: Creating high-quality unstructured meshes for finite element analysis (FEA).  
  2. **Geographic Information Systems (GIS)**: Partitioning regions based on proximity to landmarks (e.g., service area mapping).  
  3. **Numerical Simulation**: Defining computational domains for fluid dynamics or heat transfer models.


Prospective Users
=================

The toolkit serves three primary user groups, each with distinct usage patterns:
  1. **Computational Geometry Developers**: Integrate the C++ core module into high-performance applications (e.g., mesh generators, simulation engines) to leverage efficient diagram construction and querying.
  2. **Data Analysts & GIS Data Analysis**: Use the Python API or command-line tool to process point data (e.g., GPS coordinates) and generate Voronoi-based partitions, with built-in visualization for quick insights.
  3. **Students & Researchers**: Experiment with Voronoi diagram properties via the intuitive Python interface, or modify the C++ algorithm core to test custom optimizations.


System Architecture
===================

Input/Output

- **Input**: 2D point sets in three formats:
    - TXT file (one point per line: `x y`).
    - Numpy array.
    - C++ `std::vector` of `Point` structs.
- **Output**:
    - Structured Voronoi diagram data (JSON file or in-memory objects) containing point indices, vertex coordinates, and edge associations.
    - Matplotlib-generated visualization (points + Voronoi edges).
    - Nearest neighbor query results (generator point ID for target points).


Modularization
  The system is divided into two decoupled layers:
    1. **C++ Core Layer**: Implements Fortune’s algorithm, data structures (VoronoiCell, Point, Edge), and grid index.
    2. **Python Wrapper Layer**: Uses pybind11 to expose core C++ functionality as a Python module.


Constraints
  - Limited to **2D Euclidean space**.
  - Supports generator point sets of size 1000–10000+ (tested up to 50000 points).
  - Does not support weighted or power Voronoi diagrams (focus on standard 2D Voronoi).


API Description
===============

High-level Python API description

.. code:: python=

    import numpy as np
    from voronoi_toolkit import Voronoi
    
    # --- Class Definition ---
    class Voronoi:
        """A class to compute and interact with a Voronoi diagram."""
    
        def __init__(self, points: np.ndarray | list[tuple[float, float]]):
            """Builds the diagram upon instantiation."""
            # ... implementation ...
    
        def find_nearest(self, target_points: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
            """Finds the nearest generator point index for one or more target points."""
            # ... implementation ...
    
        def plot(self, show_points=True, show_edges=True, ax=None):
            """Visualizes the Voronoi diagram using Matplotlib."""
            # ... implementation ...
    
        @property
        def cells(self) -> list[dict]:
            """Returns a list of dictionaries, each describing a Voronoi cell."""
            # ... implementation ...
    
        def save_json(self, filepath: str):
            """Saves the diagram data to a JSON file."""
            # ... implementation ...
    
        @staticmethod
        def from_txt(filepath: str) -> 'Voronoi':
            """Creates a Voronoi instance from a TXT file."""
            # ... implementation ...

Example usage:

.. code:: python=

    import numpy as np
    import voronoi_toolkit
    
    # 1. Build from a NumPy array
    points = np.random.rand(50, 2) * 100  # 50 random points
    vd = voronoi_toolkit.Voronoi(points)
    
    # 2. Find the nearest generator for a set of target points
    targets = np.array([[25, 25], [50, 50]])
    nearest_indices = vd.find_nearest(targets)
    print(f"Nearest generator indices: {nearest_indices}")
    
    # 3. Visualize the diagram
    vd.plot()
    
    # 4. Save the diagram structure to a file
    vd.save_json("diagram_output.json")


Engineering Infrastructure
==========================

1. Automatic Build System - CMake

2. Lisence - We use Apache 2.0 lisencing for this project!

3. Testing Framework
    - **C++ Unit Tests**: Use **Google Test** to validate core functionality:
        - **Golden Standard Tests**: Compare output against precomputed correct Voronoi diagrams for 3 test cases (uniform, boundary-concentrated, random point sets).
        - **Edge Case Tests**: Validate handling of duplicate points, collinear points, and empty input.
        - **Performance Tests**: Benchmark build/query time for 1000–10000+ points.
    - **Python Tests**: Use `pytest` to verify API consistency and CLI functionality.

4. Documentation
    - **User Documentation**: `README.md` (installation, quick start, API examples).
    - **Code Documentation (We put function docs in codebase! while high-level docs in README.md, or other README.md in various entries)**:
        - C++: Doxygen-style comments for classes/methods (e.g., `/** Build Voronoi diagram... */`).
        - Python: Docstrings for all API functions (e.g., `def build(points: List[Tuple[float, float]]): """Build diagram from points..."""`).


Schedule
========

* Week 1 (09/22): Build a prototype for fortune algorithms. Identify the core algorithms.
* Week 2 (09/29): Develop C++ testcases for the fortune algorithm code.
* Week 3 (10/06): Redesign and construct the prototype code and make it neat and meet our needs(eg. visualization). w/tests
* Week 4 (10/27): Redesign and construct the prototype code and make it neat and meet our needs(eg. visualization). w/tests
* Week 5 (11/03): Implement multi-threaded optimizations (parallel preprocessing, divide-and-conquer build)
* Week 6 (11/10): Implement multi-threaded optimizations (parallel preprocessing, divide-and-conquer build)
* Week 7 (11/17): Run performance benchmarks; fix edge cases (collinear points, large coordinates); complete `README.md` and API documentation.
* Week 8 (11/24): Flextime, Integrate all modules; run end-to-end tests; tag v1.0 release on GitHub.



References
==========

1. Fortune, S. (1987). A sweepline algorithm for Voronoi diagrams. *Algorithmica*, 2(1-4), 153–174.
2. pybind11 Documentation. https://pybind11.readthedocs.io/en/stable/
3. CMake Documentation. https://cmake.org/cmake/help/latest/
