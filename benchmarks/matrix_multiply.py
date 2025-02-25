import torch
import time
import pandas as pd
from typing import Tuple, Dict, Optional
from densparse import DenSparseMatrix
from densparse.mapping_utils import square_cycle_mapping

# Check if MPS is available
MPS_AVAILABLE = torch.backends.mps.is_available()
CPU_DEVICE = torch.device("cpu")
MPS_DEVICE = torch.device("mps") if MPS_AVAILABLE else CPU_DEVICE

# Check if MKL is available
MKL_AVAILABLE = torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else False

def create_random_sparse_matrix(size: int, density: int) -> Tuple[
    Dict[str, DenSparseMatrix], 
    Dict[str, torch.Tensor], 
    Dict[str, Dict[str, Optional[torch.Tensor]]]
]:
    """Create random sparse matrix using different implementations."""
    results = {
        'densparse': {},
        'dense': {},
        'sparse': {'cpu': {}, 'mps': {}}
    }
    
    # Create mapping
    mapping = square_cycle_mapping(size, size, density)
    
    # Create CPU DenSparse matrix
    densparse_cpu = DenSparseMatrix(mapping, max_batch=size)
    densparse_cpu.randomize_weights()
    results['densparse']['cpu'] = densparse_cpu
    
    # Create dense matrix from DenSparse (CPU)
    dense_cpu = densparse_cpu.to_dense()
    results['dense']['cpu'] = dense_cpu
    
    # Create basic sparse CPU formats
    results['sparse']['cpu']['coo'] = dense_cpu.to_sparse()
    results['sparse']['cpu']['csr'] = dense_cpu.to_sparse_csr()
    results['sparse']['cpu']['csc'] = dense_cpu.to_sparse_csc()
    
    # Create MKL-dependent formats only if available
    if MKL_AVAILABLE:
        try:
            results['sparse']['cpu']['bsr'] = dense_cpu.to_sparse_bsr(blocksize=min(size, 4))
        except Exception as e:
            print(f"BSR not supported for size {size}: {e}")
            results['sparse']['cpu']['bsr'] = None
            
        try:
            results['sparse']['cpu']['bsc'] = dense_cpu.to_sparse_bsc(blocksize=min(size, 4))
        except Exception as e:
            print(f"BSC not supported for size {size}: {e}")
            results['sparse']['cpu']['bsc'] = None
    
    # Create MPS versions if available
    if MPS_AVAILABLE:
        try:
            # Create new mapping and matrix for MPS
            mapping_mps = mapping.clone()
            densparse_mps = DenSparseMatrix(mapping_mps, max_batch=size)
            with torch.no_grad():
                densparse_mps.forward_weights.copy_(densparse_cpu.forward_weights)
            densparse_mps = densparse_mps.to(MPS_DEVICE)
            results['densparse']['mps'] = densparse_mps
            
            # Create dense MPS matrix
            dense_mps = dense_cpu.clone().to(MPS_DEVICE)
            results['dense']['mps'] = dense_mps
            
            # Note: Skip sparse MPS formats as they're not supported
        except Exception as e:
            print(f"Error setting up MPS tensors: {e}")
            results['densparse']['mps'] = None
            results['dense']['mps'] = None
    
    return results['densparse'], results['dense'], results['sparse']

def benchmark_matrix_multiply(sizes: range, densities: range, n_trials: int = 100) -> pd.DataFrame:
    """Benchmark matrix multiplication across implementations."""
    results = []
    
    for size in sizes:
        for density in densities:
            if density > size:
                continue
                
            print(f"Testing size={size}, density={density}", flush=True)
            
            # Create matrices
            t0 = time.perf_counter()
            densparse_mats, dense_mats, sparse_mats = create_random_sparse_matrix(size, density)
            t1 = time.perf_counter()
            init_time = t1 - t0
            
            # Create batch tensors for testing
            batch_cpu = torch.randn(size, size)
            batch_mps = batch_cpu.to(MPS_DEVICE) if MPS_AVAILABLE else None
            
            # Time multiplications
            times = {
                'densparse_cpu_x_dense': [],
                'dense_cpu_x_dense': [],
            }
            
            # Add available sparse CPU formats
            for fmt in sparse_mats['cpu']:
                if sparse_mats['cpu'][fmt] is not None:
                    times[f'sparse_cpu_{fmt}_x_dense'] = []
            
            # Add MPS implementations if available
            if MPS_AVAILABLE and batch_mps is not None:
                if densparse_mats.get('mps') is not None:
                    times['densparse_mps_x_dense'] = []
                if dense_mats.get('mps') is not None:
                    times['dense_mps_x_dense'] = []
            
            for _ in range(n_trials):
                # CPU implementations
                t0 = time.perf_counter()
                _ = densparse_mats['cpu'] @ batch_cpu
                times['densparse_cpu_x_dense'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                _ = dense_mats['cpu'] @ batch_cpu
                times['dense_cpu_x_dense'].append(time.perf_counter() - t0)
                
                # CPU sparse formats
                for fmt in sparse_mats['cpu']:
                    if sparse_mats['cpu'][fmt] is not None:
                        try:
                            t0 = time.perf_counter()
                            _ = sparse_mats['cpu'][fmt] @ batch_cpu
                            times[f'sparse_cpu_{fmt}_x_dense'].append(time.perf_counter() - t0)
                        except Exception as e:
                            print(f"CPU {fmt} multiplication failed: {e}")
                
                # MPS implementations
                if MPS_AVAILABLE and batch_mps is not None:
                    if densparse_mats.get('mps') is not None:
                        torch.mps.synchronize()
                        t0 = time.perf_counter()
                        _ = densparse_mats['mps'] @ batch_mps
                        torch.mps.synchronize()
                        times['densparse_mps_x_dense'].append(time.perf_counter() - t0)
                    
                    if dense_mats.get('mps') is not None:
                        torch.mps.synchronize()
                        t0 = time.perf_counter()
                        _ = dense_mats['mps'] @ batch_mps
                        torch.mps.synchronize()
                        times['dense_mps_x_dense'].append(time.perf_counter() - t0)
            
            # Record average times
            results.append({
                'size': size,
                'density': density,
                'init_time': init_time,
                **{k: sum(v)/max(len(v), 1) if len(v) > 0 else float('nan') for k,v in times.items()}
            })
            
    return pd.DataFrame(results)

MAX_POWER = 14 # 2^14 = 16384
MAX_POWER = 8

if __name__ == '__main__':
    # Print system info
    print(f"MPS available: {MPS_AVAILABLE}")
    print(f"MKL available: {MKL_AVAILABLE}")
    if MPS_AVAILABLE:
        print(f"MPS device: {MPS_DEVICE}")
        print(f"PyTorch version: {torch.__version__}")
    
    sizes = [2**i for i in range(1, MAX_POWER + 1)]
    densities = [2**i for i in range(0, MAX_POWER + 1)]
    
    results = benchmark_matrix_multiply(sizes, densities)
    results.to_csv('benchmark_results.csv', index=False)
    
    # Print summary
    print("\nAverage times by implementation:")
    for col in results.columns:
        if col not in ['size', 'density', 'init_time']:
            avg = results[col].mean()
            if not pd.isna(avg):
                print(f"{col}: {avg:.6f}s")
            else:
                print(f"{col}: Not available") 