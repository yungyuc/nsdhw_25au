#!/usr/bin/env python3
import _matrix
import time

# Matrix size
size = 1000

# Create matrices
A = _matrix.Matrix(size, size)
B = _matrix.Matrix(size, size)

# Initialize matrices
print(f"Initializing {size}x{size} matrices...")
for i in range(size):
    for j in range(size):
        A[i, j] = float(i + j)
        B[i, j] = float(i - j + 1)

print(f'\nMatrix size: {size}x{size}')

# Time naive multiplication
print('\nTiming naive multiplication...')
start = time.time()
C_naive = _matrix.multiply_naive(A, B)
time_naive = time.time() - start
print(f'Naive time: {time_naive:.4f} seconds')

# Time tiled multiplication
print('\nTiming tiled multiplication...')
start = time.time()
C_tile = _matrix.multiply_tile(A, B, 64)
time_tile = time.time() - start
print(f'Tiled time: {time_tile:.4f} seconds')

# Time MKL multiplication
print('\nTiming MKL multiplication...')
start = time.time()
C_mkl = _matrix.multiply_mkl(A, B)
time_mkl = time.time() - start
print(f'MKL time: {time_mkl:.4f} seconds')

# Performance comparison
print('\nPerformance comparison:')
print(f'Speedup (tile vs naive): {time_naive / time_tile:.2f}x')
print(f'Speedup (mkl vs naive): {time_naive / time_mkl:.2f}x')
print(f'Tile is {(1 - time_tile/time_naive)*100:.1f}% faster than naive')

# Verify results
print('\nVerifying results match...')
assert C_naive.equals(C_tile, 1e-8), 'Naive and tile results do not match'
assert C_naive.equals(C_mkl, 1e-8), 'Naive and MKL results do not match'
print('All results match!')

# Write performance report
with open('performance.txt', 'w') as f:
    f.write('Matrix Multiplication Performance Report\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'Matrix size: {size}x{size}\n\n')
    f.write('Timing Results:\n')
    f.write('-' * 50 + '\n')
    f.write(f'Naive multiplication:  {time_naive:.4f} seconds\n')
    f.write(f'Tiled multiplication:  {time_tile:.4f} seconds\n')
    f.write(f'MKL DGEMM:             {time_mkl:.4f} seconds\n\n')
    f.write('Performance Comparison:\n')
    f.write('-' * 50 + '\n')
    f.write(f'Speedup (tile vs naive): {time_naive / time_tile:.2f}x\n')
    f.write(f'Speedup (mkl vs naive):  {time_naive / time_mkl:.2f}x\n\n')
    f.write(f'Tile is {(1 - time_tile/time_naive)*100:.1f}% faster than naive\n')
    f.write(f'MKL is {(1 - time_mkl/time_naive)*100:.1f}% faster than naive\n\n')
    f.write('Verification: All results match (within tolerance 1e-8)\n')

print('\nPerformance report saved to performance.txt')
