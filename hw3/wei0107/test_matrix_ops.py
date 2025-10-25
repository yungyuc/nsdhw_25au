import numpy as np
import pytest
import _matrix as mo

rtol = 1e-10
atol = 1e-10

def rand_mat(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))

def toM(a):  # numpy -> C++ Matrix
    return mo.Matrix.from_numpy(a.astype(np.double, copy=True))

def assert_close(A, B):
    assert A.shape == B.shape
    assert np.allclose(A, B, rtol=rtol, atol=atol)

def test_matrix_basic():
    m = mo.Matrix(2,3)
    a = np.array(m, copy=False)
    a[:] = 2.0
    b = np.array(m, copy=False)
    assert np.allclose(b, 2.0)

def test_naive_small():
    A = rand_mat(5,7,1)
    B = rand_mat(7,4,2)
    C_ref = A @ B
    C = mo.multiply_naive(toM(A), toM(B)).to_numpy()
    assert_close(C, C_ref)

def test_tiled_matches_naive():
    A = rand_mat(32,32,3)
    B = rand_mat(32,32,4)
    C_naive = mo.multiply_naive(toM(A), toM(B)).to_numpy()
    C_tiled = mo.multiply_tile(toM(A), toM(B), tile=16).to_numpy()
    assert_close(C_tiled, C_naive)

@pytest.mark.parametrize("tile",[16,32,64,128])
def test_tiled_matches_numpy(tile):
    A = rand_mat(33,31,5)
    B = rand_mat(31,29,6)
    C_ref = A @ B
    C = mo.multiply_tile(toM(A), toM(B), tile=tile).to_numpy()
    assert_close(C, C_ref)

def test_mkl_matches_numpy_if_available():
    A = rand_mat(17,19,7)
    B = rand_mat(19,13,8)
    C_ref = A @ B
    try:
        C = mo.multiply_mkl(toM(A), toM(B)).to_numpy()
    except Exception:
        pytest.skip("DGEMM unavailable")
        return
    assert_close(C, C_ref)
