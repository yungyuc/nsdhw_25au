====================
R-Tree Search Engine
====================

Basic Information
=================

**Repository URL**: https://github.com/heyvrann/RTSE

This project builds a spatial search engine based on R-trees. 
It indexes points, segments, and boxes in 2D/3D, 
supports dynamic insert/remove/update with C++ core 
and Python interface.

Problem to Solve
================

The project addresses the need for scalable, 
low-latency spatial search over large 2D/3D geometric datasets. 
As the data grow and queries become frequent, 
a linear scan is no longer viable for domains 
such as robotic collision and game or simulation workloads. 
We therefore focus on a spatial index that maintains stable performance 
under mixed streams of updates and queries.

Our approach employs an R-tree whose nodes store minimum bounding rectangles, 
enabling sub-linear range searches by pruning irrelevant branches 
of the hierarchy. 
The indexed entities include points, segments, and boxes, 
each bound to an integer identifier for application-level bookkeeping. 
Because real-world datasets are dynamic, the index must support online insert, 
remove, and move operations. 
For large initial datasets, bulk loading is used to shorten build time 
and improve query quality. 
The end goal is to deliver reproducible spatial queries through 
a clear C++/Python API that integrates easily 
into the target application domains.

Prospective Users
=================

The intended users are engineers and researchers 
who issue frequent spatial queries on 2D/3D data, such as:

* GIS and mapping services.
* Robotics and autonomous systems.
* scientific computing and simulation.

In all cases, the workflow is the same: insert geometries 
with IDs through a compact C++/Python API, 
then issue window or distance to obtain ID sets that are consumed 
by the host application for visualization, planning, analysis, or statistics.

System Architecture
===================

**Overview**

* C++ R-tree spaial index with nodes storing **MBR**s; 
mirrored Python bindings expose the same API.
* 2D ``point/segment/box``.
* 3D ``point/box`` (optional).

**Inputs / Outputs**

* Input: ``(Geometry, id)`` pairs from CSV/WKT or in-memory streams.
* Output: list of **ID**s; the host app materializes objects by ID.

**Constraints & Assumptions**

* Axis-aligned window queries only.
* Minimal returns (IDs, optional distances) 
so applications control materialization and downstream work.
* Future extensions possible: polygons, orientaed boxes, concurrent quries

**Modularization**

* ``core``: geometry and R-tree wrapper.
* ``query engine``: range strategies, pruning, priority queues.
* ``I/O layer``: CSV/WKT/in-memory ingestion; optional serialization.
* ``binding``: pybind11 layer exposing the API to Python.
* ``tests``: correctness vs. brute force and latency measurement.

API Description
===============

**Basic geometric**
   
1. ``Point2``: :math:`(x, y)`.
2. ``Box2``: :math:`(x_{min}, y_{min}, x_{max}, y_{max})`.
3. ``Segment2``: :math:`((x_{1}, y_{1}), (x_{2}, y_{2}))`.

In real implementation, ``Box2`` and ``Segment2`` 
will be instantiated by two ``Point2``.

**API**

1. ``insert``: add ``(geometry, id)`` to the index and ``id`` should be unique.
2. ``erase``: remove by ``id``.
3. ``update``: replace geometry for an existing ``id``.
4. ``query_range``: axis-aligned window search 
then returns matching ``id``s (order not guaranteed).


Engineering Infrastructure
==========================

* **Build system**: CMake (for C++), Pybind11 (for Python).

* **Version control**: Git + GitHub.

* **Testing**: ``Catch2`` or ``GoogleTest`` (for C++), ``pytest`` (for Python).

* **Documentation**: 

1. Sphinx (reStructuredText)
2. Doxygen + Breathe

Schedule
========

* Planning phase (8 weeks from 09/22 to 11/16):
* Week 1 (09/22): Set up repository and project skeleton. Study R-tree 
data structure. Add a minimal CI on GitHub Actions to run one smoke test.
* Week 2 (09/29): Wire up CMake, pybind11, and implement R-tree in C++.
* Week 3 (10/06): Finish R-tree implementation. Expand CI to execute the 
initial unit tests.
* Week 4 (10/13): API implementation for both C++ and Python. Add unit tests 
with a brute-force oracle.
* Week 5 (10/20): Introduce property-based tests (Hypothesis) 
and establish baseline benchmarks (build time, per-query latency, QPS).
* Week 6 (10/27): Scale experiments. Start documentation draft 
(Basic Information, Problem, Users, Architecture, API).
* Week 7 (11/3): Enforce quality gates and make CI required for PRs. 
Finalize benchmark plots and reproducible scripts. Prepare demo notes.
* Week 8 (11/10): Publish documentation and rehearse the presentation.

References
==========

1. Boost.Geometry documentation: 
https://www.boost.org/doc/libs/1_77_0/libs/geometry/doc/html/index.html
2. Guttman, A. (1984). 
*R-trees: A dynamic index structure for spatial searching.*
3. Pybind11 documentation: https://pybind11.readthedocs.io
