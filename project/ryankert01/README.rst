=================
fastVoronoi
=================

Basic Information
=================

- A high-performance Voronoi diagram toolkit designed for applied spatial
  analytics and seamless integration into modern GIS workflows(GeoPandas). 
- **GitHub Repository URL**:
  https://github.com/ryankert01/fastVoronoi
- The toolkit provides both low-level C++ APIs (for performance) and high-level
  Python APIs (for usability).
- **A high-performance C++/Python toolkit** for 2D Voronoi diagram construction
  and nearest neighbor querying, supporting computational - geometry scenarios
  like mesh generation. Also, I'm planning to support visualization in
  fewer-point scenarios(fewer than 100 points).


Problem to Solve & Prospective Users
================

This toolkit is designed for `Data Analysts, GIS Professionals`, and Data
Scientists who need to translate raw location data into actionable spatial
insights. The ideal user is focused on practical applications, such as defining
service areas for businesses, analyzing resource allocation for public services,
or modeling market territories. They require tools that integrate seamlessly
into existing geospatial workflows, operate within specific geographic
boundaries (like a city limit), and produce context-rich visualizations for
immediate interpretation.

The toolkit bridges the gap between high-performance computational geometry and
applied spatial analytics by speaking the user's language: GeoPandas DataFrames
and bounded polygons.


System Architecture
===================

Input/Output

- **Input**: 2D point sets in these formats:
    - TXT file (one point per line: `x y`).
    - Numpy array.
    - geoPandas dataframe (`gpd.GeoDataFrame`)
- **Output**:
    - Structured Voronoi diagram data (in-memory objects)
      containing point indices, vertex coordinates, and edge associations.
    - Matplotlib-generated visualization.
    - (Optional) Visualize based on a real backgroud map (w/ contextily)


Modularization
  The system is divided into two decoupled layers:
    1. **C++ Core Layer**: Implements Fortune's algorithm, data structures
       (VoronoiCell, Point, Edge), and grid index.
    2. **Python Wrapper Layer**: Uses pybind11 to expose core C++ functionality
       as a Python module.


Constraints
  - Limited to **2D Euclidean space**.
  - Supports generator point sets of size up to 10000 (tested up to 20000
    points).
  - Does not support weighted or power Voronoi diagrams (focus on standard 2D
    Voronoi).


API Description
===============

High-level Python API description

.. code:: python=

    # --- Class Definition ---
    class Voronoi:
        """A class to compute bounded Voronoi diagrams for geospatial analysis."""

        def __init__(self, 
                    points: gpd.GeoDataFrame | np.ndarray | list[tuple[float, float]], 
                    boundary: Polygon | None = None):
            """
            Builds the diagram, clipping it to an optional boundary.

            If points is a GeoDataFrame, its 'geometry' column is used.
            """ 
            # ... implementation ...

        def to_geodataframe(self) -> gpd.GeoDataFrame:
            """
            Converts the Voronoi cells into a GeoDataFrame.

            Each row represents a cell, with its geometry and an ID linking
            back to the original generator point.
            """
            # ... implementation ...

        def plot(self, with_basemap: bool = False, ax=None):
            """
            Visualizes the Voronoi diagram, optionally on a map background.

            If with_basemap is True, it overlays the diagram on a tile map
            (e.g., OpenStreetMap) for spatial context.

            ax: axis to draw graph
            """ 
            # ... implementation ...
        
        def find_nearest(self, 
                        target_points: np.ndarray | list[tuple[float, float]]
                        ) -> np.ndarray:
            """
            Finds the nearest generator point index for one or more target points.
            """ 
            # ... implementation ...


Example usage:

.. code:: python=

    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    import fastvoronoi

    # 1. Create geospatial data from locations of interest 🗺️
    stations_data = {
        'name': ['Station A', 'Station B', 'Station C'],
        'geometry': [Point(1, 5), Point(3, 1), Point(8, 6)]
    }
    stations_gdf = gpd.GeoDataFrame(stations_data, crs="EPSG:4326")

    # 2. Define a city boundary as an Area of Interest (AOI)
    city_boundary = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])

    # 3. Build the bounded Voronoi diagram directly from the GeoDataFrame
    fv = fastvoronoi.Voronoi(stations_gdf, boundary=city_boundary)

    # 4. Visualize with a real map background for immediate context 📍
    fv.plot(with_basemap=True)


    # 5. Get the result as a GeoDataFrame for further spatial analysis
    # This output is perfect for calculating zone areas or joining with other data.
    service_areas_gdf = fv.to_geodataframe()
    print(service_areas_gdf)


Engineering Infrastructure
==========================

1. Automatic Build System - CMake

2. Lisence - We use Apache 2.0 lisencing for this project!

3. Testing Framework
    - **C++ Unit Tests**: Use **Google Test** to validate core functionality:
        - **Golden Standard Tests**: Compare output against precomputed correct
          Voronoi diagrams for 3 test cases (uniform, boundary-concentrated,
          random point sets).
        - **Edge Case Tests**: Validate handling of duplicate points, coœllinear
          points, and empty input.
        - **Performance Tests**: Benchmark build/query time for 1000–10000+
          points.
    - **Python Tests**: Use `pytest` to verify API consistency and CLI
      functionality.

4. Documentation
    - **User Documentation**: `README.md` (installation, quick start, API
      examples).
    - **Code Documentation (We put function docs in codebase! while high-level
      docs in README.md, or other README.md in various entries)**:
        - C++: Doxygen-style comments for classes/methods (e.g., `/** Build
          Voronoi diagram... */`).
        - Python: Docstrings for all API functions (e.g., `def build(points:
          List[Tuple[float, float]]): """Build diagram from points..."""`).


Schedule
========

* Week 1 (09/22): Build a prototype for fortune algorithms and data structures.
  Identify the core algorithms.
* Week 2 (09/29): Keep building prototype and develop simple C++ testcases.
* Week 3 (10/06): Redesign and construct the prototyped code(algorithm and data
  structures) and make it neat. w/tests
* Week 4 (10/27): Redesign and construct the prototyped code(algorithm and data
  structures) and make it neat. w/tests
* Week 5 (11/03): Implement visualizations
* Week 6 (11/10): Implement multi-threaded optimizations (parallel
  preprocessing, divide-and-conquer build)
* Week 7 (11/17): Implement multi-threaded optimizations (parallel
  preprocessing, divide-and-conquer build)
* Week 8 (11/24): Run performance benchmarks; fix edge cases (collinear points,
  large coordinates); complete `README.md` and API documentation. Tag 1.0
  release on Github.



References
==========

1. Fortune, S. (1986). A sweepline algorithm for Voronoi diagrams. https://doi.org/10.1145/10515.1054
2. pybind11 Documentation. https://pybind11.readthedocs.io/en/stable/
3. CMake Documentation. https://cmake.org/cmake/help/latest/
