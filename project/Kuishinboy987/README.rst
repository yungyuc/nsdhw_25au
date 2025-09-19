=================
R-Tree
=================

Basic Information
=================

GitHub Repo: https://github.com/Kuishinboy987/R-tree

About: An R-tree implementation for 2D (or possibly 3D) data in an efficient 
coding style, including search, insert, and delete operations.

Problem to Solve
================

As with other tree data structures, R-trees are primarily used for search.
However, unlike most trees, R-trees also encode spatial information about objects.
This allows us to efficiently solve problems of indexing multi-dimensional objects, 
such as "Find all restaurants in a certain area" or "Find the nearest gas station."

Like all tree data structures, an R-tree is composed of nodes (root, internal 
nodes, and leaves) and rules (storage and operations) that define how the 
nodes are managed. Therefore, to implement R-trees effectively, we only need to 
understand these rules and optimize the structure in computer memory.

Prospective Users
=================

R-trees are highly effective for spatial access. Because they adapt to variations 
in data, they can be applied in many fields, such as databases, computer graphics, 
games, navigation, and robotics.

System Architecture
===================

Workflow : 

1. Initialization: Create an empty R-tree(or build a tree).

2. Structure management: Insert or delete spatial objects, adjusting nodes as needed.

3. Query Execution: Traverse only the nodes whose MBR* intersects the query region.

4. Output: Return the set of results that satisfy the query.

*MBR: Minimum Bounding Rectangle, the basic concept in managing R-trees.

Constraints: 

1. After every insertion or deletion, the tree must remain balanced.

2. Node capacity (the maximum and minimum number of entries per node) is fixed.

3. Data is assumed to be static or moderately dynamic.

4. Objects are represented by rectangles (MBRs).

Modularization: 

1. Node: Defines node structure (MBR, children, pointers).

2. Insertion: Implements insertion rules and node splitting.

3. Deletion: Manages object removal and rebalancing.

4. Search: Handles spatial queries (range, nearest neighbor).

API Description
===============

One may access the api like this, 

.. code-block:: python

    import Rtree

    # initial an empty R-tree
    # for each node should have at most m, at least n children.
    mytree = Rtree(maxEntry=m, minEntry=n)

    # insert a node
    mytree.insert(rectangle, nodeID)

    # delete a certain node
    mytree.delete(rectangle, nodeID)

    # request a query - range searching
    # input: a rectangle (qurey box)
    # return: a set of search results
    mytree.search_range(query_box)

    # request a query - nearest neighbor searching
    # input: a point, or a rectangle
    # return: a set of search results
    mytree.search_nearest(qurey_box)


Engineering Infrastructure
==========================

Build system: C++: CMake ; Python: pybind11

Version control: prefer GitHub(or probably git)

Schedule
========

development of the project takes 8 weeks:

* Planning phase (9/22~): Research for the detailed knowledge about R-tree.
* Week 1 (10/20): Set up node structure.
* Week 2 (10/27): Adding Insertion module.
* Week 3 (11/03): Adjusting Insertion module with testing.
* Week 4 (11/10): Adding Deletion module.
* Week 5 (11/17): Adjusting Deletion module with testing.
* Week 6 (11/24): Adding Search module.
* Week 7 (12/01): Testing (build a large R-Tree and execute some queries)
* Week 8 (12/08): Final polishing, including preparing presentation/demo.

References
==========

1. R-tree wiki: https://en.wikipedia.org/wiki/R-tree

2. Py
