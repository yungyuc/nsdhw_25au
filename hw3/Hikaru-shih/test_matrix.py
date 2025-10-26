# test_matrix.py
import pytest
import _matrix as M

# ---------- helpers ----------
def mat_from_fn(nr, nc, f):
    """Create Matrix(nr,nc) and fill by f(i,j)."""
    A = M.Matrix(nr, nc)
    for i in range(nr):
        for j in range(nc):
            A[i, j] = f(i, j)
    return A

def nearly_equal(A, B, tol=1e-9):
    # prefer the C++ approx_equal method if it exists
    try:
        return A.approx_equal(B, tol)
    except AttributeError:
        return A == B

# ---------- basic Matrix API ----------
def test_matrix_set_get_and_shapes():
    A = M.Matrix(2, 3)
    assert A.nrow == 2 and A.ncol == 3

    # set/get
    A[0, 0] = 1.5
    A[1, 2] = -2.0
    assert A[0, 0] == pytest.approx(1.5)
    assert A[1, 2] == pytest.approx(-2.0)

def test_approx_equal_tolerance():
    A = mat_from_fn(2, 2, lambda i, j: 1.0)
    B = mat_from_fn(2, 2, lambda i, j: 1.0 + (1e-10 if (i == 0 and j == 0) else 0.0))
    assert A.approx_equal(B, 1e-9)   # within tol
    assert not A.approx_equal(B, 1e-12)

# ---------- multiplication correctness ----------
@pytest.mark.parametrize("m,k,n", [
    (1, 1, 1),
    (2, 3, 2),
    (3, 3, 3),
    (4, 2, 5),
])
def test_naive_vs_tiled_and_mkl(m, k, n):
    # A(i,j) = i + j, B(i,j) = 1 if i==j else 0 (identity-ish when square)
    A = mat_from_fn(m, k, lambda i, j: float(i + j + 1))
    B = mat_from_fn(k, n, lambda i, j: 1.0 if i == j else 0.5)  # 非奇異且避免太簡單

    C_ref = M.multiply_naive(A, B)
    C_tile = M.multiply_tile(A, B)          # default block=64
    C_mkl  = M.multiply_mkl(A, B)

    assert nearly_equal(C_ref, C_tile)
    assert nearly_equal(C_ref, C_mkl)

def test_identity_behavior_square():
    n = 4
    I = mat_from_fn(n, n, lambda i, j: 1.0 if i == j else 0.0)
    A = mat_from_fn(n, n, lambda i, j: float(i * n + j))
    C = M.multiply_naive(A, I)
    assert nearly_equal(C, A)
    C2 = M.multiply_tile(I, A)
    assert nearly_equal(C2, A)

def test_dimension_mismatch_raises():
    A = M.Matrix(2, 3)
    B = M.Matrix(4, 1)
    with pytest.raises(ValueError):
        _ = M.multiply_naive(A, B)
    with pytest.raises(ValueError):
        _ = M.multiply_tile(A, B)
    with pytest.raises(ValueError):
        _ = M.multiply_mkl(A, B)
