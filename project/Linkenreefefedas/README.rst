Contiguous N-Dimensional Array
==============================

A compact C++11/Python library for cache-friendly N-dimensional arrays with struct support and zero-copy NumPy interoperability.

Basic Information
-----------------
- GitHub Repository: https://github.com/Linkenreefefedas/Contiguous_N-Dimensional_Array.git
- About: A lightweight C++11/Python library that provides contiguous multi-dimensional arrays with clean indexing, zero-copy NumPy interoperability, and support for both fundamental and composite (struct) types.

Problem to Solve
----------------
In scientific and numerical software, multi-dimensional arrays are fundamental data structures. 
However, existing approaches in C++ and Python interoperation expose several critical issues:

1. **Complex indexing in C++**
 Raw pointer arithmetic makes multi-dimensional access cryptic and error-prone.  
2. **Performance and memory overhead** 
 Data exchange often requires redundant copies that waste memory and slow performance.  
3. **Lack of composite type support** 
 Storing multiple variables per grid point needs AoS/SoA layouts, which lightweight libraries rarely provide.  
4. **Unclear API design** 
 Users expect NumPy-like clean syntax, but C++ APIs often expose cumbersome low-level details.

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
 1. Create a new array in C++ (owning) or pass an existing NumPy array (view).
 2. The core stores/aliases the contiguous buffer and exposes safe indexing.
 3. Perform operations via C++ methods or Python wrappers.
 4. Return results as C++ values or NumPy arrays, with explicit copy control.

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

**C++ core**
::
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build --output-on-failure

**Python binding (after pybind11 lands)**
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

- Week 1: Initialize repository and CMake; build minimal `ContiguousND<float>` with shape/strides and basic tests.  
- Week 2: Extend to multiple scalar types; add clean indexing via `operator()` with error handling.  
- Week 3: Implement pybind11 bindings; enable NumPy interop with zero-copy validation and pytest.  
- Week 4: Strengthen zero-copy safety (ownership, capsule deleter); add explicit copy path and debug bounds checks.  
- Week 5: Demonstrate POD AoS usage with examples; run micro-benchmarks and refine API.  
- Week 6: Improve documentation and tutorials.  
- Week 7: Freeze v0.1 API; finalize comprehensive tests and cross-platform validation.  
- Week 8: Polish documentation; release v0.1.0 and deliver presentation/demo.

References
----------
- https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
- https://numpy.org/doc/stable/reference/arrays.interface.html