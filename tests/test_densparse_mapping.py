import torch
import pytest
from densparse import DenSparseMatrix, DenSparseMapping
from densparse.mapping import DenSparseMapping
from densparse.mapping_utils import square_cycle_mapping, up_cycle_mapping, down_cycle_mapping


# FIXTURES

@pytest.fixture
def down_asymmetric_mapping():
    """Create a 10x5 mapping with down-cycle connection pattern."""
    return down_cycle_mapping(10, 5, 3)

@pytest.fixture
def up_asymmetric_mapping():
    """Create a 5x10 mapping with up-cycle connection pattern."""
    return up_cycle_mapping(5, 10, 3)

@pytest.fixture
def manual_mapping():
    """Create a 10x5 mapping using an explicit mask matrix."""
    mask = torch.zeros(5, 10, dtype=torch.bool)

    # Create same pattern as asymmetric_mapping but with explicit mask
    for in_idx in range(10):
        base_out = in_idx // 2
        mask[base_out, in_idx] = True
        mask[(base_out - 1) % 5, in_idx] = True
        mask[(base_out + 1) % 5, in_idx] = True

    return DenSparseMapping.from_mask(mask)


# TESTS

@pytest.mark.parametrize("mapping_fixture", ["down_asymmetric_mapping", "manual_mapping"])
def test_input_connections(mapping_fixture, request):
    """Test that each input row has exactly three unmasked connections."""
    mapping = request.getfixturevalue(mapping_fixture)
    matrix = DenSparseMatrix(mapping)

    mapping_width = mapping.mapping_width
    half_width = mapping_width // 2
    for row in range(10):
        unmasked_count = matrix.forward_mask[row, :].sum().item()
        assert unmasked_count == half_width, f"Row {row} has {unmasked_count} unmasked connections, expected {half_width}"
        if mapping_fixture == "down_asymmetric_mapping":
            # Verify the correct half is unmasked
            if row % 2 == 0:
                # First three should be unmasked
                assert matrix.forward_mask[row, :half_width].all()
                assert not matrix.forward_mask[row, half_width:].any()
            else:
                # Last three should be unmasked
                assert not matrix.forward_mask[row, :half_width].any()
                assert matrix.forward_mask[row, half_width:].all()


@pytest.mark.parametrize("mapping_fixture", ["down_asymmetric_mapping", "manual_mapping"])
def test_output_connections(mapping_fixture, request):
    """Test that no outputs have masked connections."""
    mapping = request.getfixturevalue(mapping_fixture)
    matrix = DenSparseMatrix(mapping)

    mapping_width = mapping.mapping_width
    # Check that outputs 0-4 have unmasked connections
    for out_idx in range(5):
        unmasked_count = matrix.reverse_mask[out_idx].sum().item()
        assert unmasked_count == mapping_width, f"Output {out_idx} has {unmasked_count} unmasked connections, expected {mapping_width}"
        assert matrix.reverse_mask[out_idx].any(), f"Output {out_idx} has no unmasked connections"


@pytest.mark.parametrize("mapping_fixture", ["down_asymmetric_mapping", "manual_mapping"])
def test_bidirectional_mapping(mapping_fixture, request):
    """Test that all unmasked connections are bidirectional."""
    mapping = request.getfixturevalue(mapping_fixture)
    matrix = DenSparseMatrix(mapping)

    mapping_width = mapping.mapping_width
    # Check forward to reverse
    for in_idx in range(10):
        for fw_idx in range(mapping_width):
            if matrix.forward_mask[in_idx, fw_idx]:
                out_idx = matrix.forward_mapping[in_idx, fw_idx].item()
                assert matrix.reverse_mask[out_idx, fw_idx], f"No reverse mapping found for {in_idx}->{out_idx}, column {fw_idx}"
                assert matrix.reverse_mapping[out_idx, fw_idx].item() == in_idx, f"Reverse mapping mismatch for {in_idx}->{out_idx}, column {fw_idx}"

    # Check reverse to forward
    for out_idx in range(5):
        for rev_idx in range(mapping_width):
            if matrix.reverse_mask[out_idx, rev_idx]:
                in_idx = matrix.reverse_mapping[out_idx, rev_idx].item()
                assert matrix.forward_mask[in_idx, rev_idx], f"No forward mapping found for {in_idx}->{out_idx}, column {rev_idx}"
                assert matrix.forward_mapping[in_idx, rev_idx].item() == out_idx, f"Forward mapping mismatch for {in_idx}->{out_idx}, column {rev_idx}"


def test_random_mask_mappings():
    """Test that from_mask and to_dense are inverses for random masks."""
    # Test different matrix sizes
    sizes = [(5, 10), (10, 5), (20, 20), (8, 15)]

    for out_size, in_size in sizes:
        # Generate random mask with ~30% connections
        mask = torch.rand(out_size, in_size) < 0.3

        # Create mapping and convert back to dense
        mapping = DenSparseMapping.from_mask(mask)
        recovered_mask = mapping.to_dense()

        assert torch.equal(mask, recovered_mask), \
            f"Mapping failed to preserve connections for size {out_size}x{in_size}"

        # Test transpose
        transposed_mask = mask.t()
        transposed_mapping = DenSparseMapping.from_mask(transposed_mask)
        recovered_transposed = transposed_mapping.to_dense()

        assert torch.equal(transposed_mask, recovered_transposed), \
            f"Transposed mapping failed for size {out_size}x{in_size}"

        # Test that mapping.transpose() gives same result as from_mask(mask.t())
        direct_transpose = mapping.transpose()
        assert torch.equal(direct_transpose.to_dense(), transposed_mapping.to_dense()), \
            f"Direct transpose doesn't match from_mask(transpose) for size {out_size}x{in_size}"


def test_asymmetric_mapping_equivalence(down_asymmetric_mapping, manual_mapping):
    """Test that different mappings of the same connectivity pattern are equivalent."""

    # Verify both mappings produce the same connectivity
    assert torch.equal(down_asymmetric_mapping.to_dense(), manual_mapping.to_dense()), \
        "Function-based and mask-based mappings differ"

    # Verify their transposes also match
    assert torch.equal(
        down_asymmetric_mapping.transpose().to_dense(),
        manual_mapping.transpose().to_dense()
    ), "Transposed mappings differ"


def test_invalid_bidirectional_mapping():
    """Test that creating a mapping with invalid bidirectional connections raises an error."""
    def invalid_map(input_idx: int, weight_idx: int) -> tuple[int, bool]:
        # This mapping tries to map multiple inputs to the same output column
        if input_idx < 2:  # Both input 0 and 1 try to map to output 0
            return (0, True)
        return (input_idx, False)

    with pytest.raises(ValueError, match=r"Multiple inputs .* mapping to output 0 in column 0"):
        DenSparseMapping.from_function(5, 5, 2, invalid_map)


@pytest.mark.parametrize("mapping_fixture", ["down_asymmetric_mapping", "up_asymmetric_mapping", "manual_mapping"])
def test_mapping_index_validity(mapping_fixture, request):
    """Test that mapping indices are valid for safe scatter operations."""
    mapping = request.getfixturevalue(mapping_fixture)
    max_size = max(mapping.input_size, mapping.output_size)

    sets = [
        ('input', mapping.input_mapping, mapping.input_mask, mapping.input_size),
        ('output', mapping.output_mapping, mapping.output_mask, mapping.output_size),
    ]
    LABEL = 0
    MAPPING = 1
    MASK = 2
    SIZE = 3
    
    for toggle in range(2):
        for col in range(mapping.mapping_width):
            set_a = sets[toggle]
            set_b = sets[1 - toggle]
            indices = set_a[MAPPING][:, col]
            print(set_a[MAPPING][:, col])
            print(set_a[MASK][:, col])
            print(set_b[MAPPING][:, col])
            print(set_b[MASK][:, col])
            # All indices should be within valid range
            assert (indices >= 0).all() and (indices < max_size).all(), \
                f"{set_a[LABEL]} mapping indices in column {col} outside valid range [0, {max_size-1}]"
            
            # All indices in column should be unique
            assert len(indices) == len(indices.unique()), \
                f"Found duplicate indices in {set_a[LABEL]}_mapping column {col}"
            
            # Active indices should map to active outputs
            active_indices = indices[set_a[MASK][:, col]]
            for idx in active_indices:
                assert set_b[MASK][idx, col], \
                    f"{set_a[LABEL]} maps to {idx} but {set_b[LABEL]}_mask[{idx},{col}] is False"
            
            # Inactive indices should map to inactive outputs or out of range
            inactive_indices = indices[~set_a[MASK][:, col]]
            for idx in inactive_indices:
                if idx < set_b[SIZE]:
                    assert not set_b[MASK][idx, col], \
                        f"Inactive {set_a[LABEL]} maps to active {set_b[LABEL]} at {idx} in column {col}"
