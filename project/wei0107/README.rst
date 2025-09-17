PixFoundry Proposal
===================

Basic Information
-----------------

GitHub Repository: https://github.com/wei0107/pixfoundry

About:
A lightweight image processing toolkit with a C++ core and a Python interface. 
PixFoundry provides everyday photo editing features—filters,
tone and color adjustments, sharpening and visual effects, and basic
geometric transformations—all implemented with clear and extensible
numerical methods, with parallelization options for accelerated
computation.

Problem to Solve
----------------

Everyday photo editing often requires applying filters, tone adjustments,
sharpening, and simple geometric operations. While existing libraries such
as OpenCV provide powerful functionality, they are heavyweight, complex to
learn, and not tailored for coursework demonstration of numerical methods
and parallelization.

This project aims to provide a lightweight toolkit that:

- Implements common filters (average, Gaussian, median, bilateral) using
  convolution and statistical operators.
- Supports tone and color transformations with linear and nonlinear
  numerical methods.
- Provides sharpening and visual effects (e.g., emboss, cartoon).
- Offers geometric transformations such as resize, rotate, and crop.
- Explores performance through parallelization (multi-threads, SIMD).

Prospective Users
-----------------

- Students in numerical software development or image processing who need a
  compact, educational, and performance-aware image processing package.
- Researchers or hobbyists who want to test photo filtering and enhancement
  algorithms in a simple environment.
- Everyday users who want to apply lightweight editing to their photos
  without heavy software dependencies.

System Architecture
-------------------

Workflow:

1. Input: Load image as an array (NumPy in Python, raw buffer in C++).
2. Processing: Apply selected filters or operations (implemented in C++,
   optionally parallelized).
3. Output: Save the processed image or return it as an array.

Raw Memory Buffer Sketch:

.. code-block:: cpp

   // image.hpp
   #pragma once
   #include <vector>
   #include <cstddef>
   #include <cstdint>

   namespace pf {

   template <class T>
   class Image {
   public:
     Image() = default;

     Image(int h, int w, int c)
       : h_(h), w_(w), c_(c),
         stride_bytes_(w * c * int(sizeof(T))),
         buffer_(std::size_t(h) * std::size_t(w) * std::size_t(c)) {}

     T* data() noexcept { return buffer_.data(); }
     const T* data() const noexcept { return buffer_.data(); }

     inline T& at(int y, int x, int c = 0) noexcept {
       return buffer_[(y * w_ + x) * c_ + c];
     }

     int width() const noexcept { return w_; }
     int height() const noexcept { return h_; }
     int channels() const noexcept { return c_; }
     int stride_bytes() const noexcept { return stride_bytes_; }

   private:
     int h_ = 0, w_ = 0, c_ = 1;
     int stride_bytes_ = 0;
     std::vector<T> buffer_;
   };

   } // namespace pf

Constraints:

- Focus on CPU-based processing.
- Parallelization via thread-level parallelism; SIMD optimizations are
  optional.
- Limited to 2D images.

Modularization:

- **Core (C++)**: convolution, filtering, edge detection, interpolation.
- **Bindings (Python/pybind11)**: Python API exposing C++ functions.
- **Application**: command-line interface and example scripts.

API Description
---------------

Python Example:

.. code-block:: python

   import pixfoundry as pf

   img = pf.load_image("input.jpg")
   blurred = pf.gaussian_filter(img, sigma=1.6, backend="openmp")
   gray = pf.to_grayscale(img)
   sharp = pf.sharpen(img)
   resized = pf.resize(img, width=640, height=480)
   pf.save_image("output.jpg", resized)

Core APIs:

- ``gaussian_filter(img, sigma, backend="auto", border="reflect")``
- ``median_filter(img, ksize)``
- ``bilateral_filter(img, sigma_space, sigma_color)``
- ``to_grayscale(img)``, ``sepia(img)``, ``invert(img)``
- ``adjust_brightness_contrast(img, alpha, beta)``,
  ``gamma_correct(img, gamma)``
- ``sharpen(img)``, ``emboss(img)``, ``cartoonize(img, ...)``
- ``resize(img, width, height, method="bilinear")``,
  ``rotate(img, angle_deg)``, ``flip(img, axis)``,
  ``crop(img, x, y, w, h)``

Engineering Infrastructure
--------------------------

- **Build system**: CMake for C++ core; pybind11 for Python bindings.
- **Version control**: GitHub with feature branches and pull requests.
- **Continuous Integration**: GitHub Actions for build and test automation.

Schedule
--------

Development timeline (8 weeks):

- **Week 1 (10/06)**: Repository setup, project skeleton (C++ core + Python
  binding). Implement basic image I/O (load/save).
- **Week 2 (10/13)**: Implement convolution framework. Add Average and
  Gaussian filters with unit tests.
- **Week 3 (10/20)**: Implement Median and Bilateral filters. Test with
  noisy images.
- **Week 4 (10/27)**: Implement color and tone adjustments (Grayscale,
  Sepia, Invert, Brightness/Contrast).
- **Week 5 (11/03)**: Implement sharpening and effects (Sharpen, Emboss).
  Begin work on Cartoon effect (smoothing + edge detection).
- **Week 6 (11/10)**: Implement geometric operations (Resize, Rotate, Flip,
  Crop). Add OpenMP parallelization for convolution.
- **Week 7 (11/17)**: Expand testing framework (pytest + Catch2). Add demo
  examples with everyday photos. Draft documentation. Set up CI pipeline
  (GitHub Actions) for automated build and test execution.
- **Week 8 (11/24)**: Final polishing: functional tests, performance
  benchmarks (single-thread vs OpenMP), and prepare presentation/demo.

References
----------

- OpenCV Documentation: https://docs.opencv.org/
- pybind11 Documentation: https://pybind11.readthedocs.io/
- NumPy Documentation: https://numpy.org/doc/
