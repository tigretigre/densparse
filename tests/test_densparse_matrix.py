import torch
import pytest
from densparse import DenSparseMapping, DenSparseMatrix
from densparse.mapping_utils import square_cycle_mapping, up_cycle_mapping, down_cycle_mapping

@pytest.fixture
def cyclic_mapping_20():
    """Create a mapping for a 20x20 matrix with ±2 cyclic connectivity pattern."""
    return square_cycle_mapping(20, 20, 5)

@pytest.fixture
def mapping_20_to_10():
    """Create a mapping for a 10x20 matrix (10 outputs, 20 inputs)."""
    return down_cycle_mapping(20, 10, 3)

@pytest.fixture
def mapping_10_to_20():
    """Create a mapping for a 20x10 matrix (20 outputs, 10 inputs)."""
    return up_cycle_mapping(10, 20, 6)

def test_matrix_multiply_dense_vector(cyclic_mapping_20):
    sparse_mat = DenSparseMatrix(cyclic_mapping_20)

    # Create equivalent dense matrix
    dense_mat = torch.zeros(20, 20)

    # Set some test weights
    test_weights = [
        (0, 1, 0.5),
        (1, 2, 0.3),
        (19, 0, 0.4),
    ]

    for in_idx, out_idx, weight in test_weights:
        sparse_mat.set_weight(in_idx, out_idx, weight)
        dense_mat[out_idx, in_idx] = weight

    # Test multiplication with vector (input_size,)
    x = torch.randn(20)  # Single vector
    sparse_result = sparse_mat @ x
    dense_result = dense_mat @ x

    assert torch.allclose(sparse_result, dense_result, atol=1e-6)

def test_matrix_multiply_dense_matrix(cyclic_mapping_20):
    sparse_mat = DenSparseMatrix(cyclic_mapping_20)

    # Set random weights where connections exist
    with torch.no_grad():
        for out_idx in range(sparse_mat.output_size):
            for in_idx in range(sparse_mat.input_size):
                weight = torch.randn(1).item()
                sparse_mat.set_weight(in_idx, out_idx, weight, ignore_unmapped=True)

    dense_mat = sparse_mat.to_dense()

    # Test multiplication with matrix (batch_size, input_size)
    X = torch.randn(32, 20)  # Batch of 32 vectors
    sparse_result = sparse_mat @ X
    dense_result = X @ dense_mat.t()  # Note: dense_mat is (output_size, input_size)

    assert torch.allclose(sparse_result, dense_result, atol=1e-6)

def test_matrix_transpose(mapping_20_to_10):
    # Create a 20x10 matrix and its transpose
    sparse_mat = DenSparseMatrix(mapping_20_to_10)
    
    # Set random weights
    with torch.no_grad():
        for out_idx in range(sparse_mat.output_size):
            for in_idx in range(sparse_mat.input_size):
                weight = torch.randn(1).item()
                sparse_mat.set_weight(in_idx, out_idx, weight, ignore_unmapped=True)
    
    dense_mat = sparse_mat.to_dense()
    sparse_transpose = sparse_mat.transpose()
    dense_transpose = dense_mat.t()

    # Test multiplication with vector using transposed matrices
    x = torch.randn(10)
    
    # Print cell-by-cell multiplication for dense transpose
    for i in range(dense_transpose.shape[0]):  # 20 rows
        row_sum = 0
        for j in range(dense_transpose.shape[1]):  # 10 cols
            product = dense_transpose[i,j] * x[j]
            row_sum += product
    
    # Compare with sparse result
    sparse_result = sparse_transpose @ x
    dense_result = dense_transpose @ x
    assert torch.allclose(sparse_result, dense_result, atol=1e-6)

def test_matrix_multiply_rectangular(mapping_20_to_10, mapping_10_to_20):
    # Test multiplication between 20x10 and 10x20 matrices
    sparse_mat1 = DenSparseMatrix(mapping_20_to_10)
    sparse_mat2 = DenSparseMatrix(mapping_10_to_20)

    # Set random weights
    with torch.no_grad():
        for out_idx in range(sparse_mat1.output_size):
            for in_idx in range(sparse_mat1.input_size):
                weight = torch.randn(1).item()
                sparse_mat1.set_weight(in_idx, out_idx, weight, ignore_unmapped=True)

        for out_idx in range(sparse_mat2.output_size):
            for in_idx in range(sparse_mat2.input_size):
                weight = torch.randn(1).item()
                sparse_mat2.set_weight(in_idx, out_idx, weight, ignore_unmapped=True)

    dense_mat1 = sparse_mat1.to_dense()
    dense_mat2 = sparse_mat2.to_dense()
    # Test matrix chain multiplication (should result in 20x20 matrices)
    x = torch.randn(20)
    sparse_result = sparse_mat2 @ (sparse_mat1 @ x)
    dense_result = dense_mat2 @ (dense_mat1 @ x)

    assert torch.allclose(sparse_result, dense_result, atol=1e-6)

def test_matrix_gradients(cyclic_mapping_20):
    sparse_mat = DenSparseMatrix(cyclic_mapping_20)

    # Set some test weights
    test_weights = [
        (0, 1, 0.5),
        (1, 2, 0.3),
        (19, 0, 0.4),
    ]

    for in_idx, out_idx, weight in test_weights:
        sparse_mat.set_weight(in_idx, out_idx, weight)

    dense_mat = sparse_mat.to_dense().detach().requires_grad_()
    
    # Forward pass
    x = torch.randn(20, requires_grad=True)
    sparse_result = sparse_mat @ x
    dense_result = dense_mat @ x
    
    # Backward pass
    grad_output = torch.randn_like(sparse_result)
    sparse_result.backward(grad_output)
    dense_result.backward(grad_output)
    
    # Compare gradients only where connections are allowed by the mapping
    sparse_grads = sparse_mat.get_grad_matrix()
    dense_grads = dense_mat.grad
    mask = sparse_mat.mapping.to_dense()
    assert torch.allclose(sparse_grads[mask], dense_grads[mask], atol=1e-6)

def test_random_weight_initialization():
    """Test random weight initialization."""
    mapping = square_cycle_mapping(10, 10, 3)
    matrix = DenSparseMatrix(mapping)
    
    # Should start with zero weights
    assert torch.all(matrix.forward_weights == 0)
    
    # Randomize weights
    matrix.randomize_weights()
    
    # Weights should be non-zero
    assert not torch.all(matrix.forward_weights == 0)
    
    # Forward and reverse weights should match through the mapping
    dense = matrix.to_dense()
    assert torch.allclose(
        dense,
        matrix.transpose().to_dense().t(),
        atol=1e-6
    )

def test_from_dense():
    """Test creating DenSparseMatrix from dense matrix."""
    # Create a dense matrix with known pattern
    dense = torch.zeros(10, 20)
    test_weights = [
        (0, 1, 0.5),
        (1, 2, 0.3),
        (19, 0, 0.4),
        (5, 5, 0.7),
        (10, 8, -0.2),
    ]
    for in_idx, out_idx, weight in test_weights:
        dense[out_idx, in_idx] = weight
    
    # Convert to DenSparseMatrix
    sparse = DenSparseMatrix.from_dense(dense)
    
    # Check dimensions
    assert sparse.input_size == 20
    assert sparse.output_size == 10
    
    # Check weights were copied correctly
    for in_idx, out_idx, weight in test_weights:
        assert torch.allclose(torch.tensor(sparse.get_weight(in_idx, out_idx)), 
                            torch.tensor(weight), atol=1e-6)
    
    # Check zero weights weren't copied
    assert sparse.get_weight(0, 0) == 0.0
    assert sparse.get_weight(3, 3) == 0.0
    
    # Check dense conversion matches original
    converted = sparse.to_dense()
    assert torch.allclose(dense, converted)
    
    # Test multiplication gives same result
    x = torch.randn(20)
    assert torch.allclose(sparse @ x, dense @ x)
