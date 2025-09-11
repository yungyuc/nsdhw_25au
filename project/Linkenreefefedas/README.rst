CNDA
==============================

A compact C++11/Python library for cache-friendly N-dimensional arrays with struct support and zero-copy NumPy interoperability.

Basic Information
-----------------
- GitHub Repository: https://github.com/Linkenreefefedas/CNDA.git
- About: A lightweight C++11/Python library that provides contiguous multi-dimensional arrays with clean indexing, zero-copy NumPy interoperability, and support for both fundamental and composite (struct) types.
- Footprint:
   - Core (C++11): header-only, no external dependencies beyond the standard library.
   - Interop (pybind11): compiled Python extension built against pybind11 and NumPy headers.

Comparison: SimpleArray vs CNDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :widths: 18 36 36 36

   * - Aspect
     - `SimpleArray <https://github.com/solvcon/modmesh/blob/master/tests/test_buffer.py>`_
     - CNDA (proposal)
     - Key difference
   * - Scope / Goal
     - Container with many numerical ops such as sum, mean, var, std, sort, argsort, broadcast, and ghost
     - Minimal container with a clear C++/NumPy interop policy and a minimal AoS bridge
     - CNDA narrows scope and focuses on interop
   * - Memory layout
     - Row-major contiguous, supports reshape and transpose
     - Row-major contiguous, prefers zero-copy with explicit copy fallback
     - CNDA emphasizes a policy-driven choice between sharing and copying
   * - Indexing API
     - Flat and multi-dimensional. Supports slicing, ellipsis, broadcasting
     - NumPy-like ``operator()``. Slicing and broadcasting are deferred in v0.1
     - CNDA keeps indexing simple in v0.1
   * - Dtype coverage
     - Booleans, integers, floats, and complex numbers
     - POD only in v0.1 (f32/f64/i32/i64)
     - CNDA limits types to stabilize interop
   * - Composite types (AoS/SoA)
     - No first-class AoS/SoA or mapping to structured dtypes
     - Minimal AoS with NumPy structured dtype (field order/size/alignment validated)
     - CNDA adds a bridge between AoS and structured dtypes
   * - Ghost zone
     - Built-in nghost semantics and checks
     - Not supported in v0.1
     - CNDA defers ghost zones to upper layers
   * - Numerical calculators
     - Many built-in calculators, including some SIMD variants
     - None in v0.1. Use NumPy for computation
     - CNDA stays lightweight and relies on NumPy
   * - Transformations
     - Reshape and transpose. Can update in-place or return new view
     - Returns NumPy view or copy via ``to_numpy(copy=...)``
     - CNDA focuses on interop rather than rich transforms
   * - Interop policy
     - Shares with NumPy, described across APIs/tests
     - Explicit policy: zero-copy only if dtype, layout, and lifetime match. Otherwise error or ``copy=True``
     - CNDA makes sharing and copying rules explicit and predictable
   * - Errors & safety
     - Raises errors on mismatches, out-of-range access, or invalid ghost settings
     - Consistent ``TypeError``, ``ValueError``, ``RuntimeError`` with clear messages
     - CNDA standardizes error classes and messages
   * - SIMD / performance
     - Some SIMD-optimized calculators
     - Not a v0.1 goal
     - Different focus: ops in SimpleArray vs interop in CNDA
   * - Build footprint
     - Python-facing bindings
     - Core is header-only. Interop is a compiled pybind11 extension
     - CNDA minimizes dependencies with a clean split between core and interop

Problem to Solve
----------------
In scientific and numerical software, multi-dimensional arrays are fundamental data structures. 
However, existing approaches in C++ and Python interoperation expose several critical issues:

1. **Complex indexing in C++**
 Raw pointer arithmetic makes multi-dimensional access cryptic and error-prone.  
2. **Performance and memory overhead** 
 Data exchange often requires redundant copies that waste memory and slow performance.  
3. **Lack of composite type support** 
 Storing multiple values per grid point often needs AoS/SoA layouts, but most C++ array libraries lack built-in support or NumPy interoperability.  
4. **Hard-to-use indexing & copy-policy ambiguity** 
 In C++ one often writes manual stride math like `i*stride0 + j*stride1` for indexing, and it is not always clear when arrays share memory or are copied. 

Prospective Users
-----------------
Users who need a lightweight and efficient way to manage multi-dimensional arrays across C++ and Python, with minimal memory overhead.

System Architecture
-------------------
The system consists of two main layers:

1. **Core (C++11)**
     - `cnda::ContiguousND<T>` manages an owning, row-major contiguous buffer.
     - Tracks `shape` and `strides` for O(1) offset computation.
     - Clean element access via `operator()` instead of manual pointer math.
     - Supports fundamental POD types (float, double, int32, int64) and a POD AoS demo.

2. **Interop (pybind11)**
     - `from_numpy(arr, copy: bool = False)` and `to_numpy(copy: bool = False)`.
     - Prefers zero-copy when dtype/layout/lifetime are compatible.
     - With `copy=True`, performs explicit copying; otherwise, raises a clear error.

**Inputs**
 - Python: an existing `numpy.ndarray` or a desired shape.
 - C++: a shape vector (e.g., `{nx, ny, nz}`).

**Outputs**
 - C++: element references and raw pointers through the API.
 - Python: NumPy views of the same buffer (no copy if safe) or copies when requested.

**Workflow**
 1. **Python → C++**
     - A NumPy ``ndarray`` is passed into ``from_numpy(copy=...)``.
     - Interop validates dtype, alignment, and layout:
        - If compatible → returns a zero-copy view in C++.
        - If incompatible → raises an error or copies if ``copy=True``.
     - The array becomes available as a ``ContiguousND<T>`` for C++ computations.

 2. **C++ → Python**
     - A new ``ContiguousND<T>`` is allocated in C++ and filled with values.
     - Results are exported via ``to_numpy(copy=...)``:
        - If ``copy=False`` and safe → Python receives a NumPy view of the same buffer.
        - Otherwise → Python receives a copy, ensuring safety and compatibility.

**Constraints (v0.1)**
 - Row-major contiguous layout only.
 - POD element types (`float`, `double`, `int32`, `int64`).
 - Single-threaded semantics.
 - No slicing/broadcasting (reserved for later versions).
 - Structs: trivial POD AoS demo only; SoA is future work.

API Description
---------------

- **C++11 core**: templated container ``cnda::ContiguousND<T>`` for contiguous N-D arrays with explicit ``shape`` / ``strides`` and O(1) index computation.

- **Python binding (pybind11)**: module ``cnda`` with
  ``from_numpy(arr, copy: bool = False)`` (NumPy → C++ view/copy) and ``to_numpy(copy: bool = False)`` (C++ → NumPy view/copy), both defaulting to zero-copy when safe.

C++ API (namespace ``cnda``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Primary container (header prototype)**

.. code-block:: cpp

  // contiguous_nd.hpp
  #pragma once
  #include <vector>
  #include <cstddef>
  #include <initializer_list>

  namespace cnda {

  template<class T>
  class ContiguousND {
  public:
    // Construct an owning, row-major contiguous buffer of given shape.
    explicit ContiguousND(std::vector<std::size_t> shape);

    // Basic introspection.
    const std::vector<std::size_t>& shape()   const noexcept;
    const std::vector<std::size_t>& strides() const noexcept;
    std::size_t ndim()  const noexcept;
    std::size_t size()  const noexcept;

    // Raw access.
    T*       data()       noexcept;
    const T* data() const noexcept;

    // Indexing helpers (O(1) offset).
    std::size_t index(std::initializer_list<std::size_t> idx) const;
    T& operator()(std::size_t i);
    T& operator()(std::size_t i, std::size_t j);
    T& operator()(std::size_t i, std::size_t j, std::size_t k);
    // (Variadic overloads may be added later.)
  };

  } // namespace cnda

**Minimal usage (prototype)**

.. code-block:: cpp

  #include "contiguous_nd.hpp"
  #include <iostream>
  using cnda::ContiguousND;

  int main() {
    ContiguousND<float> a({3, 4});   // 3x4 contiguous (row-major)
    a(1, 2) = 42.0f;
    std::cout << "a(1,2) = " << a(1,2) << "\\n";
    std::cout << a.ndim() << "D, size=" << a.size() << "\\n";
    return 0;
  }

Python API (module ``cnda``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Top-level functions & types**

``from_numpy(arr: numpy.ndarray, copy: bool = False) -> ContiguousND_*``

- Returns a zero-copy view if the dtype and layout are compatible.
- If not compatible:
   - With ``copy=True``: performs an explicit copy.
   - With ``copy=False``: raises ``ValueError`` or ``TypeError`` on the Python side.
- The dtype-specific suffix for ``ContiguousND_*`` is one of: ``f32``, ``f64``, ``i32``, ``i64``.

``ContiguousND_*.to_numpy(copy: bool = False) -> numpy.ndarray``

- By default (``copy=False``), returns a NumPy view (no copy).
- With ``copy=True``, returns a new array, isolating lifetime/ownership from the C++ object.

**Round-trip example (zero-copy)**

.. code-block:: python

  import numpy as np
  import cnda

  # NumPy → C++ view (no copy)
  x = np.arange(12, dtype=np.float32).reshape(3, 4)
  a = cnda.from_numpy(x, copy=False)  # strict zero-copy

  # C++ → NumPy view (no copy)
  y = a.to_numpy(copy=False)          # shares memory with x
  y[1, 2] = 42
  assert x[1, 2] == 42
  assert y.ctypes.data == x.ctypes.data  # same buffer

**Structured dtype (AoS) example**

.. code-block:: python

  import numpy as np, cnda

  cell_dtype = np.dtype([('u','<f4'), ('v','<f4'), ('flag','<i4')], align=True)
  arr = np.zeros((nx, ny), dtype=cell_dtype, order='C')

  a = cnda.from_numpy(arr, copy=False)  # zero-copy only if field order/size/alignment match the C++ struct
  out = a.to_numpy(copy=False)          # view when safe; use copy=True to isolate lifetime

**Allocate on C++ side and expose to NumPy**

.. code-block:: python

  import numpy as np
  import cnda

  b = cnda.ContiguousND_f32([2, 3])     # C++-owned contiguous buffer
  B = b.to_numpy(copy=False)             # NumPy view (no copy)
  B.fill(7.0)
  assert (B == 7.0).all()

  # If you need isolation from the C++ owner:
  B_copy = b.to_numpy(copy=True)         # explicit copy with independent lifetime

Zero-copy and error semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``from_numpy(arr, copy=False)`` is zero-copy only if:

1. Dtype matches the bound container type
2. Array is C-contiguous (row-major)
3. Lifetime is safe (binding keeps the producer alive)

Otherwise:

- If ``copy=True`` → make an explicit copy  
- If ``copy=False`` → raise ``ValueError``/``TypeError`` (Python) or throw ``std::invalid_argument`` (C++)

``to_numpy(copy=False)`` returns a NumPy view with a capsule deleter.  
Use ``copy=True`` to force duplication and isolate the lifetime from the C++ owner.

Bounds & safety
~~~~~~~~~~~~~~~
- `operator()` performs no bounds checking (performance-first).
- Provide `at(...)` or a Debug flag (e.g., `-DCNDA_BOUNDS_CHECK=ON`) to enable bounds checks in development.

Threading model
~~~~~~~~~~~~~~~
- v0.1 semantics are single-threaded.
- Concurrent read-only access may be safe if the producer lifetime is guaranteed; concurrent writes require external synchronization and are out of scope for v0.1.

Exceptions and error types
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python layer: `TypeError` (dtype mismatch), `ValueError` (layout/size incompatibility), `RuntimeError` (lifetime/capsule issues).
- C++ layer: throws `std::invalid_argument` or `std::runtime_error` with clear messages.

Engineering Infrastructure
--------------------------

Automatic build
~~~~~~~~~~~~~~~
Prereqs: CMake (>=3.18), C++11 compiler, Python 3.9+.

**C++ core** (header-only; build here is only for tests and examples)
::
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build --output-on-failure

**Python binding** (requires pybind11 and NumPy headers)
::
  python -m venv .venv
  # Windows: .\.venv\Scripts\activate
  # Linux/macOS:
  source .venv/bin/activate
  pip install -U pip
  pip install -e .

Version control
~~~~~~~~~~~~~~~
- GitHub public repo; default branch: ``main`` (protected).
- Conventional commits (``feat:``, ``fix:``, ``test:``, ``docs:``, ``chore:``).
- Issues/Milestones aligned to the 8-week schedule.

Testing
~~~~~~~
- C++: Catch2 via CTest (shape/strides/index; negative cases).
- Python: pytest with NumPy as oracle; zero-copy checks via ``ctypes.data``; dtype/contiguity validation.

Documentation
~~~~~~~~~~~~~
- ``README.rst`` = proposal + quickstart; updated via PRs.
- ``docs/`` for zero-copy policy, ownership rules, API examples.

Schedule
--------
8-week plan; Weeks 1–6 focus on core; Weeks 7–8 on integration/delivery.

- Week 1 (10/20) : Initialize repository and CMake; build minimal `ContiguousND<float>` with shape/strides and basic tests.  
- Week 2 (10/27) : Extend to multiple scalar types; add clean indexing via `operator()` with error handling.  
- Week 3 (11/3) : Implement pybind11 bindings; enable NumPy interop with zero-copy validation and pytest.  
- Week 4 (11/10) : Strengthen zero-copy safety (ownership, capsule deleter); add explicit copy path and debug bounds checks.  
- Week 5 (11/17) : Demonstrate POD AoS usage with examples; run micro-benchmarks and refine API.  
- Week 6 (11/24) : Improve documentation and tutorials.  
- Week 7 (12/1) : Freeze v0.1 API; finalize comprehensive tests and cross-platform validation.  
- Week 8 (12/8) : Polish documentation; release v0.1.0 and deliver presentation/demo.

References
----------
- https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
- https://numpy.org/doc/stable/reference/arrays.interface.html