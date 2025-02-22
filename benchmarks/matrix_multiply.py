import torch
import time
import pandas as pd
from typing import Tuple, Dict
from densparse import DenSparseMatrix, DenSparseMapping
from densparse.mapping_utils import square_cycle_mapping

def create_random_sparse_matrix(size: int, density: int) -> Tuple[DenSparseMatrix, torch.Tensor, torch.Tensor]:
    """Create random sparse matrix using different implementations.
    
    Args:
        size: Matrix dimension (size x size)
        density: Number of connections per node in cyclic pattern
        
    Returns:
        (densparse_mat, dense_mat, sparse_mat)
    """
    # Create DenSparse matrix
    mapping = square_cycle_mapping(size, size, density)
    densparse = DenSparseMatrix(mapping, max_batch=size)
    densparse.randomize_weights()
    
    # Convert to dense
    dense = densparse.to_dense()
    
    # Convert to PyTorch sparse
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1]]
    sparse = torch.sparse_coo_tensor(indices, values, (size, size))
    
    return densparse, dense, sparse

def benchmark_matrix_multiply(sizes: range, densities: range, n_trials: int = 100) -> pd.DataFrame:
    """Benchmark matrix multiplication across implementations.
    
    Args:
        sizes: Range of matrix dimensions to test
        densities: Range of densities to test for each size
        n_trials: Number of multiplication trials to average
        
    Returns:
        DataFrame with timing results
    """
    results = []
    
    for size in sizes:
        for density in densities:
            if density > size:
                continue
                
            print(f"Testing size={size}, density={density}")
            
            # Create matrices
            t0 = time.perf_counter()
            mat1_ds, mat1_dense, mat1_sparse = create_random_sparse_matrix(size, density)
            t1 = time.perf_counter()
            init_time = t1 - t0
            
            # Create dense matrix for sparse x dense test
            dense_batch = torch.randn(size, size)
            
            # Time multiplications
            times = {
                'densparse_x_dense': [],
                'dense_x_dense': [],
                'sparse_x_dense': [],
                'densparse_x_densparse': [],
                'dense_x_dense_2': [],
                'sparse_x_sparse': []
            }
            
            for _ in range(n_trials):
                # Sparse x Dense
                t0 = time.perf_counter()
                _ = mat1_ds @ dense_batch
                times['densparse_x_dense'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                _ = mat1_dense @ dense_batch
                times['dense_x_dense'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                _ = mat1_sparse @ dense_batch
                times['sparse_x_dense'].append(time.perf_counter() - t0)
                
                # Create second sparse matrix
                mat2_ds, mat2_dense, mat2_sparse = create_random_sparse_matrix(size, density)
                
                # Sparse x Sparse
                t0 = time.perf_counter()
                _ = mat1_ds @ mat2_ds
                times['densparse_x_densparse'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                _ = mat1_dense @ mat2_dense
                times['dense_x_dense_2'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                _ = mat1_sparse @ mat2_sparse
                times['sparse_x_sparse'].append(time.perf_counter() - t0)
            
            # Record average times
            results.append({
                'size': size,
                'density': density,
                'init_time': init_time,
                **{k: sum(v)/len(v) for k,v in times.items()}
            })
            
    return pd.DataFrame(results)

if __name__ == '__main__':
    sizes = [2**i for i in range(1, 14)]  # 2 to 8192
    densities = [2**i for i in range(0, 14)]  # 1 to 8192
    
    results = benchmark_matrix_multiply(sizes, densities)
    results.to_csv('benchmark_results.csv', index=False)
    
    # Print summary
    print("\nAverage times by implementation:")
    for col in results.columns:
        if col not in ['size', 'density']:
            print(f"{col}: {results[col].mean():.6f}s") 