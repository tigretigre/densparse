import torch
import pytest
from densparse.mapping_utils import up_cycle_mapping, down_cycle_mapping, square_cycle_mapping

def test_square_cycle_mapping():
    """Test cyclic mapping for same-size layers."""
    size = 10
    max_in = 3
    mapping = square_cycle_mapping(size, size, max_in)
    dense = mapping.to_dense()

    # Each row/col should have exactly max_in connections
    assert (dense.sum(dim=1) == max_in).all()
    assert (dense.sum(dim=0) == max_in).all()

    # Check pattern - each node should connect to itself and neighbors
    for i in range(size):
        connected = dense[i].nonzero().squeeze()
        expected = torch.tensor([
            (i - 1) % size,
            i,
            (i + 1) % size,
        ])
        assert torch.equal(connected.sort()[0], expected.sort()[0])

@pytest.mark.parametrize("ratio", [
    2,   # 2x scale with 6 outputs per input
    3,   # 3x scale with 9 outputs per input
    8,  # 8x scale with 24 outputs per input
])
def test_up_cycle_mapping_ratios(ratio):
    """Test mapping from smaller to larger layer with different ratios."""
    input_size = 10
    output_size = input_size * ratio
    max_in = 3
    max_out = max_in * ratio
    
    mapping = up_cycle_mapping(input_size, output_size, max_out)
    dense = mapping.to_dense()

    # Each output connects to max_out // ratio inputs
    assert (dense.sum(dim=1) == max_out // ratio).all()
    # Each input connects to max_out outputs
    assert (dense.sum(dim=0) == max_out).all()

    # Check locality - each input maps to ratio consecutive outputs
    for i in range(input_size):
        connected = dense[:, i].nonzero().squeeze()
        base = i * ratio
        width = max_out
        offset = (max_in - 1) // 2 * ratio # Mappings centered on the input
        expected = torch.tensor([
            (base - offset + i) % output_size
            for i in range(width)
        ])
        assert torch.equal(connected, expected.sort()[0])

@pytest.mark.parametrize("ratio", [
    2,  # 2x scale with 3 outputs per input
    3,  # 3x scale with 3 outputs per input
    8,  # 8x scale with 3 outputs per input
])
def test_down_cycle_mapping_ratios(ratio):
    """Test mapping from larger to smaller layer with different ratios."""
    output_size = 10
    input_size = output_size * ratio
    max_out = 3
    
    mapping = down_cycle_mapping(input_size, output_size, max_out)
    dense = mapping.to_dense()

    # Each output connects to max_out * ratio inputs
    assert (dense.sum(dim=1) == max_out * ratio).all()
    # Each input connects to max_out outputs
    assert (dense.sum(dim=0) == max_out).all()

    # Check pattern - groups of ratio inputs map to same outputs
    for i in range(0, input_size, ratio):
        base = i // ratio
        expected = torch.tensor([
            (base - 1) % output_size,
            base,
            (base + 1) % output_size,
        ])
        # All inputs in group should connect to same outputs
        for j in range(ratio):
            connected = dense[:, i + j].nonzero().squeeze()
            assert torch.equal(connected, expected.sort()[0])

def test_invalid_ratios():
    """Test error cases for non-integer ratios."""
    with pytest.raises(ValueError, match="must be multiple"):
        up_cycle_mapping(10, 15, 3)  # 1.5x is invalid
        
    with pytest.raises(ValueError, match="must be multiple"):
        down_cycle_mapping(15, 10, 3)  # 1.5x is invalid 