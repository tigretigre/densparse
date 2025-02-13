import torch
import pytest
import torch.nn as nn
from densparse import DenSparse
from densparse.mapping_utils import square_cycle_mapping

@pytest.fixture
def layer():
    """Create a standard 20-node DenSparse layer for testing."""
    mapping = square_cycle_mapping(20, 20, 5)
    return DenSparse(mapping)

def test_densparse_shape(layer):
    dense_weights = layer.to_dense()
    assert dense_weights.shape == (20, 20)

def test_weight_setting(layer):
    layer.set_weight(0, 1, 0.5)
    assert layer.get_weight(0, 1) == 0.5
    assert layer.to_dense()[1, 0] == 0.5

    with pytest.raises(ValueError):
        layer.set_weight(0, 5, 0.5)  # Connection doesn't exist

def test_self_connected_forward_backward(layer):
    torch.manual_seed(42)
    batch_size = 4
    input_size = 20

    # Create regular fully connected layer
    fc = nn.Linear(input_size, input_size, bias=False)

    # Copy weights and zero out unused connections
    with torch.no_grad():
        dense_weights = fc.weight.data
        sparse_dense = layer.to_dense()
        dense_weights.copy_(sparse_dense)
        dense_weights.mul_(layer.matrix.mapping.to_dense())

    # Test forward pass
    x = torch.randn(batch_size, input_size)
    dense_out = fc(x)
    sparse_out = layer(x)
    assert torch.allclose(dense_out, sparse_out, atol=1e-6)

    # Test backward pass
    grad_output = torch.randn_like(dense_out)
    dense_out.backward(grad_output)
    sparse_out.backward(grad_output)

    # Compare gradients where connections exist
    mask = layer.matrix.mapping.to_dense()
    assert torch.allclose(fc.weight.grad[mask], layer.get_grad_matrix()[mask], atol=1e-6)
