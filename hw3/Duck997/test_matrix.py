import _matrix as mx

def test_shapes_and_basic():
    a = mx.Matrix(2, 3)
    b = mx.Matrix(3, 2)
    a.at(0, 0); b.at(0, 0)  # touch API
    c1 = mx.multiply_naive(a, b)
    c2 = mx.multiple_tile(a, b, 2)
    assert c1.rows() == 2 and c1.cols() == 2
    assert c2.rows() == 2 and c2.cols() == 2

def test_mkl_exists_and_matches():
    if not mx.blas_available():
        import pytest
        pytest.skip("BLAS not available; skipping MKL comparison")
    a = mx.Matrix(2, 3)
    b = mx.Matrix(3, 2)
    # fill some simple values
    for i in range(2):
        for k in range(3):
            a.set(i, k, float(i + k))
    for k in range(3):
        for j in range(2):
            b.set(k, j, float(k + j))
    c_naive = mx.multiply_naive(a, b)
    c_mkl = mx.multiply_mkl(a, b)
    assert c_naive.rows() == c_mkl.rows()
    assert c_naive.cols() == c_mkl.cols()


