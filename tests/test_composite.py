"""Tests for CompositeMapping class."""
import torch
import pytest
from densparse.matrix import DenSparseMatrix
from densparse.mapping_utils import square_cycle_mapping
from densparse.composite import CompositeMapping
from densparse.directional import DirectionalRangeCompositeMapping


class TestCompositeMapping:
    """Tests for the base CompositeMapping class."""

    def test_init_with_matrices(self):
        """CompositeMapping can be initialized with a list of matrices."""
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        
        composite = CompositeMapping([m1, m2])
        assert len(composite.matrices) == 2

    def test_init_empty(self):
        """CompositeMapping can be initialized with empty list."""
        composite = CompositeMapping([])
        assert len(composite.matrices) == 0

    def test_forward_single_matrix(self):
        """Forward with single matrix matches that matrix's output."""
        size = 8
        mapping = square_cycle_mapping(size, size, 3)
        m = DenSparseMatrix(mapping)
        m.randomize_weights()
        
        composite = CompositeMapping([m])
        
        x = torch.randn(size)
        expected = m.forward(x)
        result = composite.forward(x)
        
        assert torch.allclose(result, expected)

    def test_forward_sums_matrices(self):
        """Forward sums outputs from all matrices."""
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=2))
        m1.randomize_weights()
        m2.randomize_weights()
        
        composite = CompositeMapping([m1, m2])
        
        x = torch.randn(size)
        expected = m1.forward(x) + m2.forward(x)
        result = composite.forward(x)
        
        assert torch.allclose(result, expected)

    def test_forward_multiple_matrices(self):
        """Forward works with many matrices."""
        size = 10
        matrices = []
        for offset in range(-2, 3):
            m = DenSparseMatrix(square_cycle_mapping(size, size, 5, offset=offset))
            m.randomize_weights()
            matrices.append(m)
        
        composite = CompositeMapping(matrices)
        
        x = torch.randn(size)
        expected = sum(m.forward(x) for m in matrices)
        result = composite.forward(x)
        
        assert torch.allclose(result, expected)

    def test_forward_batch(self):
        """Forward works with batched input."""
        size = 8
        batch_size = 4
        
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        m1.randomize_weights()
        m2.randomize_weights()
        
        composite = CompositeMapping([m1, m2])
        
        x = torch.randn(batch_size, size)
        expected = m1.forward(x) + m2.forward(x)
        result = composite.forward(x)
        
        assert torch.allclose(result, expected)

    def test_forward_empty_returns_zeros(self):
        """Forward with no matrices returns zeros."""
        composite = CompositeMapping([])
        
        x = torch.randn(10)
        result = composite.forward(x)
        
        assert torch.allclose(result, torch.zeros_like(x))

    def test_forward_different_sizes_raises(self):
        """Forward with mismatched matrix sizes raises error."""
        m1 = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(10, 10, 3))
        
        composite = CompositeMapping([m1, m2])
        
        x = torch.randn(8)
        with pytest.raises((RuntimeError, ValueError)):
            composite.forward(x)

    def test_normalize_incoming_not_implemented(self):
        """Base class normalize_incoming raises NotImplementedError."""
        m = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        composite = CompositeMapping([m])
        
        with pytest.raises(NotImplementedError):
            composite.normalize_incoming()

    def test_prune_incoming_not_implemented(self):
        """Base class prune_incoming raises NotImplementedError."""
        m = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        composite = CompositeMapping([m])
        
        with pytest.raises(NotImplementedError):
            composite.prune_incoming(keep=2)

    def test_to_device(self):
        """CompositeMapping can be moved to a device."""
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        
        composite = CompositeMapping([m1, m2])
        composite = composite.to(torch.device('cpu'))
        
        # Should work without error
        x = torch.randn(size)
        result = composite.forward(x)
        assert result.shape == (size,)

    def test_to_dense_single_matrix(self):
        """to_dense with single matrix matches that matrix's to_dense."""
        size = 8
        m = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m.randomize_weights()
        
        composite = CompositeMapping([m])
        
        expected = m.to_dense()
        result = composite.to_dense()
        
        assert result.shape == (size, size)
        assert torch.allclose(result, expected)

    def test_to_dense_sums_matrices(self):
        """to_dense sums dense representations of all matrices."""
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=2))
        m1.randomize_weights()
        m2.randomize_weights()
        
        composite = CompositeMapping([m1, m2])
        
        expected = m1.to_dense() + m2.to_dense()
        result = composite.to_dense()
        
        assert torch.allclose(result, expected)

    def test_to_dense_empty_raises(self):
        """to_dense with empty composite raises ValueError."""
        composite = CompositeMapping([])
        
        with pytest.raises(ValueError, match="empty composite"):
            composite.to_dense()

    def test_to_dense_mismatched_sizes_raises(self):
        """to_dense with different size matrices raises ValueError."""
        m1 = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(10, 10, 3))
        
        composite = CompositeMapping([m1, m2])
        
        with pytest.raises(ValueError, match="Inconsistent matrix sizes"):
            composite.to_dense()

    def test_forward_matches_dense_matmul(self):
        """Forward pass result matches dense matrix multiplication."""
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        m1.randomize_weights()
        m2.randomize_weights()
        
        composite = CompositeMapping([m1, m2])
        
        x = torch.randn(size)
        
        # Composite forward
        result = composite.forward(x)
        
        # Dense matmul
        dense = composite.to_dense()
        expected = dense @ x
        
        assert torch.allclose(result, expected)

    def test_forward_batch_matches_dense_matmul(self):
        """Batched forward pass matches dense matrix multiplication."""
        size = 8
        batch_size = 4
        
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        m1.randomize_weights()
        m2.randomize_weights()
        
        composite = CompositeMapping([m1, m2])
        
        x = torch.randn(batch_size, size)
        
        # Composite forward
        result = composite.forward(x)
        
        # Dense matmul (need to transpose for batch)
        dense = composite.to_dense()
        expected = (dense @ x.T).T  # (size, size) @ (size, batch) -> (size, batch) -> (batch, size)
        
        assert torch.allclose(result, expected)


# ============================================================================
# Tests for DirectionalRangeCompositeMapping
# ============================================================================

def _build_test_matrices(N: int, S: int, D: int, offset_func=None):
    """Helper to build a 2D list of matrices for testing.
    
    Args:
        N: Neurons per slice
        S: Number of slices
        D: Max distance (number of ds columns is D+1)
        offset_func: Optional function(src_slice, dst_slice) -> offset
        
    Returns:
        2D list: matrices[src_slice][ds] -> DenSparseMatrix or None
    """
    if offset_func is None:
        offset_func = lambda s, d: 0
    
    matrices = []
    for src_slice in range(S):
        row = []
        for ds in range(D + 1):
            dst_slice = src_slice + ds
            if dst_slice >= S:
                row.append(None)
                continue
            
            # Width decreases with ds (simplified model)
            width = max(1, 2 * D + 1 - 2 * ds)
            offset = offset_func(src_slice, dst_slice)
            
            mapping = square_cycle_mapping(N, N, width, offset=offset)
            m = DenSparseMatrix(mapping)
            m.randomize_weights()
            row.append(m)
        matrices.append(row)
    return matrices


class TestDirectionalRangeCompositeMapping:
    """Tests for DirectionalRangeCompositeMapping."""

    def test_init(self):
        """Can initialize with 2D matrix list."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        assert composite._N == N
        assert composite._S == S
        assert composite._D == D

    def test_forward_matches_naive_loop(self):
        """Forward output matches naive per-matrix loop."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Input: one value per neuron across all slices
        x = torch.randn(S * N)
        
        # Naive computation: loop over all matrices
        expected = torch.zeros(S * N)
        for src_slice in range(S):
            for ds in range(D + 1):
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                dst_slice = src_slice + ds
                src_start = src_slice * N
                dst_start = dst_slice * N
                x_slice = x[src_start:src_start + N]
                out_slice = m.forward(x_slice)
                expected[dst_start:dst_start + N] += out_slice
        
        result = composite.forward(x)
        
        assert torch.allclose(result, expected, atol=1e-5)

    def test_forward_batch(self):
        """Forward works with batched input."""
        N, S, D = 8, 4, 2
        batch_size = 3
        matrices = _build_test_matrices(N, S, D)
        
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        x = torch.randn(batch_size, S * N)
        
        # Compute expected by running each batch element separately
        expected = torch.stack([
            composite.forward(x[i]) for i in range(batch_size)
        ])
        
        # Now compute using batched forward
        # Reset matrices to original state (randomize again with same seed)
        torch.manual_seed(42)
        matrices2 = _build_test_matrices(N, S, D)
        composite2 = DirectionalRangeCompositeMapping(matrices2, N, S, D)
        
        torch.manual_seed(42)
        matrices3 = _build_test_matrices(N, S, D)
        composite3 = DirectionalRangeCompositeMapping(matrices3, N, S, D)
        
        result = composite3.forward(x)
        expected2 = torch.stack([composite2.forward(x[i]) for i in range(batch_size)])
        
        assert torch.allclose(result, expected2, atol=1e-5)

    def test_forward_with_hex_offsets(self):
        """Forward works correctly with hex offset pattern."""
        N, S, D = 8, 6, 3
        
        def hex_offset(src_slice, dst_slice):
            """Hex packing offset based on parity."""
            src_even = (src_slice % 2 == 0)
            dst_even = (dst_slice % 2 == 0)
            if src_even == dst_even:
                return 0
            elif src_even:
                return 1
            else:
                return -1
        
        matrices = _build_test_matrices(N, S, D, offset_func=hex_offset)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        x = torch.randn(S * N)
        
        # Naive computation
        expected = torch.zeros(S * N)
        for src_slice in range(S):
            for ds in range(D + 1):
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                dst_slice = src_slice + ds
                src_start = src_slice * N
                dst_start = dst_slice * N
                x_slice = x[src_start:src_start + N]
                out_slice = m.forward(x_slice)
                expected[dst_start:dst_start + N] += out_slice
        
        result = composite.forward(x)
        
        assert torch.allclose(result, expected, atol=1e-5)

    def test_normalize_incoming_l1(self):
        """normalize_incoming L1-normalizes per destination."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set some non-zero weights using the setter
        for row in matrices:
            for m in row:
                if m is not None:
                    weights = torch.ones_like(m.forward_weights)
                    m.forward_weights = weights
        
        composite.normalize_incoming()
        
        # Check that incoming weights per destination sum to 1
        for dst in range(S * N):
            dst_slice = dst // N
            total = 0.0
            for src_slice in range(S):
                ds = dst_slice - src_slice
                if ds < 0 or ds > D:
                    continue
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                # Sum weights coming into this destination from this matrix
                for src_local in range(N):
                    total += m.get_weight(src_local, dst % N)
            
            if total > 0:
                assert abs(total - 1.0) < 1e-5, f"Destination {dst} weights sum to {total}"

    def test_prune_incoming(self):
        """prune_incoming keeps top-k weights per destination."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set distinguishable weights using the setter
        weight_val = 1.0
        for row in matrices:
            for m in row:
                if m is not None:
                    weights = torch.full_like(m.forward_weights, weight_val)
                    m.forward_weights = weights
                    weight_val += 1.0
        
        keep = 2
        composite.prune_incoming(keep=keep)
        
        # Check that each destination has at most `keep` non-zero incoming weights
        for dst in range(S * N):
            dst_slice = dst // N
            count = 0
            for src_slice in range(S):
                ds = dst_slice - src_slice
                if ds < 0 or ds > D:
                    continue
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                for src_local in range(N):
                    if m.get_weight(src_local, dst % N) != 0:
                        count += 1
            
            assert count <= keep, f"Destination {dst} has {count} connections, expected <= {keep}"

    def test_widths_property(self):
        """Widths are correctly inferred from matrices."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Width for each ds should match the mapping width of first matrix in that column
        for ds in range(D + 1):
            expected_width = matrices[0][ds].mapping.mapping_width if matrices[0][ds] else 0
            assert composite._widths[ds] == expected_width

    def test_to_dense_shape(self):
        """to_dense returns correct shape."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        dense = composite.to_dense()
        
        assert dense.shape == (S * N, S * N)

    def test_to_dense_block_structure(self):
        """to_dense places each matrix in correct block position."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        dense = composite.to_dense()
        
        # Check each matrix is in the right block
        for src_slice in range(S):
            for ds in range(D + 1):
                m = matrices[src_slice][ds]
                dst_slice = src_slice + ds
                if dst_slice >= S:
                    continue
                if m is None:
                    continue
                
                expected_block = m.to_dense()
                
                dst_start = dst_slice * N
                dst_end = dst_start + N
                src_start = src_slice * N
                src_end = src_start + N
                
                actual_block = dense[dst_start:dst_end, src_start:src_end]
                
                assert torch.allclose(actual_block, expected_block), \
                    f"Block ({src_slice}->{dst_slice}) mismatch"

    def test_forward_matches_dense_matmul(self):
        """Forward pass matches dense matrix multiplication."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        x = torch.randn(S * N)
        
        # Composite forward
        result = composite.forward(x)
        
        # Dense matmul
        dense = composite.to_dense()
        expected = dense @ x
        
        assert torch.allclose(result, expected, atol=1e-5)

    def test_forward_batch_matches_dense_matmul(self):
        """Batched forward matches dense matrix multiplication."""
        N, S, D = 8, 4, 2
        batch_size = 3
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        x = torch.randn(batch_size, S * N)
        
        # Composite forward
        result = composite.forward(x)
        
        # Dense matmul
        dense = composite.to_dense()
        expected = (dense @ x.T).T
        
        assert torch.allclose(result, expected, atol=1e-5)

    def test_normalize_result_matches_dense_normalize(self):
        """Normalize result: each destination row sums to 1."""
        N, S, D = 8, 4, 2
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Normalize incoming
        composite.normalize_incoming()
        
        # Get dense after normalization
        # Dense is (dst, src), so each row is a destination with all its incoming
        dense_after = composite.to_dense()
        
        # Each row with incoming connections should sum to 1
        for row in range(dense_after.shape[0]):
            row_sum = dense_after[row, :].abs().sum()
            if row_sum > 1e-8:
                assert torch.allclose(
                    row_sum,
                    torch.tensor(1.0),
                    atol=1e-5,
                ), f"Row {row} doesn't sum to 1 after normalization (got {row_sum})"

    def test_prune_result_matches_dense_verification(self):
        """Prune keeps at most k non-zeros per row (destination) when verified via dense."""
        N, S, D = 8, 4, 2
        keep = 3
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Prune
        composite.prune_incoming(keep=keep)
        
        # Get dense and verify each row (destination)
        # Dense is (dst, src), so each row is a destination
        dense = composite.to_dense()
        
        for row in range(dense.shape[0]):
            nonzero_count = (dense[row, :].abs() > 1e-8).sum().item()
            assert nonzero_count <= keep, \
                f"Row {row} has {nonzero_count} non-zeros, expected <= {keep}"


# ============================================================================
# Integration tests: DirectionalRangeCompositeMapping with cylinder topology
# ============================================================================

import math


def _compute_cylinder_connections(N: int, S: int, D: float, self_conn: bool = False):
    """Compute cylinder topology connections (reference implementation).
    
    Returns set of (src_flat, dst_flat) pairs.
    """
    connections = set()
    dy_per_slice = math.sqrt(3) / 2.0
    max_slice_offset = math.ceil(D / dy_per_slice)
    max_horiz = math.ceil(D) + 1
    
    for s_src in range(S):
        src_offset = 0.5 if s_src % 2 == 1 else 0.0
        
        for i in range(N):
            x_src = float(i) + src_offset
            y_src = s_src * dy_per_slice
            src_flat = s_src * N + i
            
            for ds in range(0, max_slice_offset + 1):
                s_dst = s_src + ds
                if s_dst >= S:
                    break
                
                dst_offset = 0.5 if s_dst % 2 == 1 else 0.0
                y_dst = s_dst * dy_per_slice
                dy = y_dst - y_src
                
                if dy > D:
                    break
                
                max_dx = math.sqrt(D * D - dy * dy) if D * D > dy * dy else 0.0
                
                for dj in range(-max_horiz, max_horiz + 1):
                    j = (i + dj) % N
                    x_dst = float(j) + dst_offset
                    
                    # Cyclic horizontal distance
                    dx = (x_dst - x_src) % N
                    if dx > N / 2.0:
                        dx -= N
                    dx = abs(dx)
                    
                    if dx > max_dx:
                        continue
                    
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > D:
                        continue
                    
                    dst_flat = s_dst * N + j
                    
                    if src_flat == dst_flat and not self_conn:
                        continue
                    
                    connections.add((src_flat, dst_flat))
    
    return connections


def _compute_width_for_ds(ds: int, D: float, N: int) -> int:
    """Compute connection width for a given slice offset (matching DensparseCompositeBackend)."""
    dy = ds * math.sqrt(3) / 2.0
    if dy > D:
        return 0
    
    max_dx = math.sqrt(D * D - dy * dy)
    
    if ds % 2 == 1:
        width = int(math.ceil(max_dx + 0.5)) * 2
    else:
        width = int(math.floor(max_dx)) * 2 + 1
    
    return min(width, N)


def _compute_hex_offset(src_slice: int, dst_slice: int) -> int:
    """Compute hex packing offset based on slice parities."""
    src_even = (src_slice % 2 == 0)
    dst_even = (dst_slice % 2 == 0)
    
    if src_even == dst_even:
        return 0
    elif src_even:
        # Even→odd: dst is shifted +0.5, pattern needs to shift left
        return -1
    else:
        # Odd→even: src is shifted +0.5, pattern needs to shift right
        return 0


def _build_cylinder_matrices(N: int, S: int, D: float):
    """Build matrices matching the cylinder topology pattern."""
    max_ds = min(math.ceil(D / (math.sqrt(3) / 2.0)), S - 1)
    
    matrices = []
    for src_slice in range(S):
        row = []
        for ds in range(max_ds + 1):
            dst_slice = src_slice + ds
            if dst_slice >= S:
                row.append(None)
                continue
            
            width = _compute_width_for_ds(ds, D, N)
            if width <= 0:
                row.append(None)
                continue
            
            # Apply hex offset for cross-parity connections
            offset = _compute_hex_offset(src_slice, dst_slice)
            mapping = square_cycle_mapping(N, N, width, offset=offset)
            m = DenSparseMatrix(mapping)
            row.append(m)
        matrices.append(row)
    
    return matrices, max_ds


class TestDirectionalRangeWithCylinderTopology:
    """Integration tests verifying DirectionalRangeCompositeMapping captures cylinder topology."""

    def test_covers_all_topology_connections(self):
        """Composite mapping covers all connections from cylinder topology."""
        N, S, D = 8, 4, 2.0
        
        # Get reference connections
        ref_connections = _compute_cylinder_connections(N, S, D)
        
        # Build composite mapping
        matrices, max_ds = _build_cylinder_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, max_ds)
        
        # Check that every reference connection is representable in the composite
        missing = []
        for src_flat, dst_flat in ref_connections:
            src_slice = src_flat // N
            dst_slice = dst_flat // N
            src_local = src_flat % N
            dst_local = dst_flat % N
            
            ds = dst_slice - src_slice
            if ds > max_ds or ds < 0:
                missing.append((src_flat, dst_flat, "ds out of range"))
                continue
            
            m = matrices[src_slice][ds]
            if m is None:
                missing.append((src_flat, dst_flat, "matrix is None"))
                continue
            
            # Check if the mapping contains this connection
            mapping = m.mapping
            found = False
            for col in range(mapping.mapping_width):
                if mapping.input_mask[src_local, col] and mapping.input_mapping[src_local, col] == dst_local:
                    found = True
                    break
            
            if not found:
                missing.append((src_flat, dst_flat, "not in mapping"))
        
        assert not missing, f"Missing connections: {missing[:10]}... ({len(missing)} total)"

    def test_forward_matches_reference(self):
        """Forward pass matches reference sparse matrix-vector multiply."""
        N, S, D = 8, 4, 2.0
        
        matrices, max_ds = _build_cylinder_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, max_ds)
        
        # Set random weights
        torch.manual_seed(123)
        for row in matrices:
            for m in row:
                if m is not None:
                    m.randomize_weights()
        
        # Input
        x = torch.randn(S * N)
        
        # Composite forward
        result = composite.forward(x)
        
        # Reference: naive per-matrix loop
        expected = torch.zeros(S * N)
        for src_slice in range(S):
            for ds in range(max_ds + 1):
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                dst_slice = src_slice + ds
                src_start = src_slice * N
                dst_start = dst_slice * N
                x_slice = x[src_start:src_start + N]
                out_slice = m.forward(x_slice)
                expected[dst_start:dst_start + N] += out_slice
        
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("N,S,D", [
        (4, 3, 1.5),
        (8, 4, 2.0),
        (16, 8, 3.0),
        (32, 16, 5.0),
    ])
    def test_various_sizes(self, N, S, D):
        """Works correctly for various topology sizes."""
        ref_connections = _compute_cylinder_connections(N, S, D)
        matrices, max_ds = _build_cylinder_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, max_ds)
        
        # Count representable connections
        representable = 0
        for src_flat, dst_flat in ref_connections:
            src_slice = src_flat // N
            dst_slice = dst_flat // N
            src_local = src_flat % N
            dst_local = dst_flat % N
            
            ds = dst_slice - src_slice
            if ds > max_ds or ds < 0:
                continue
            
            m = matrices[src_slice][ds]
            if m is None:
                continue
            
            mapping = m.mapping
            for col in range(mapping.mapping_width):
                if mapping.input_mask[src_local, col] and mapping.input_mapping[src_local, col] == dst_local:
                    representable += 1
                    break
        
        assert representable == len(ref_connections), \
            f"Only {representable}/{len(ref_connections)} connections representable for N={N}, S={S}, D={D}"

    def test_normalize_preserves_topology(self):
        """Normalization doesn't zero out valid connections."""
        N, S, D = 8, 4, 2.0
        matrices, max_ds = _build_cylinder_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, max_ds)
        
        # Set all weights to 1
        for row in matrices:
            for m in row:
                if m is not None:
                    weights = torch.ones_like(m.forward_weights)
                    m.forward_weights = weights
        
        # Normalize
        composite.normalize_incoming()
        
        # Verify forward still produces non-zero output
        x = torch.ones(S * N)
        result = composite.forward(x)
        
        # Every destination should receive some input (unless it has no incoming connections)
        # Check that most outputs are non-zero
        nonzero_count = (result.abs() > 1e-6).sum().item()
        assert nonzero_count > 0, "All outputs are zero after normalization"

    def test_prune_respects_keep_limit(self):
        """Pruning keeps at most 'keep' connections per destination."""
        N, S, D = 8, 4, 2.0
        keep = 3
        
        matrices, max_ds = _build_cylinder_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, max_ds)
        
        # Set random weights
        torch.manual_seed(456)
        for row in matrices:
            for m in row:
                if m is not None:
                    m.randomize_weights()
        
        # Prune
        composite.prune_incoming(keep=keep)
        
        # Check that each destination has at most 'keep' incoming connections
        for dst_global in range(S * N):
            dst_slice = dst_global // N
            dst_local = dst_global % N
            
            count = 0
            for src_slice in range(dst_slice + 1):
                ds = dst_slice - src_slice
                if ds > max_ds:
                    continue
                m = matrices[src_slice][ds]
                if m is None:
                    continue
                
                for src_local in range(N):
                    if abs(m.get_weight(src_local, dst_local)) > 1e-8:
                        count += 1
            
            assert count <= keep, f"Destination {dst_global} has {count} connections, expected <= {keep}"


# ============================================================================
# Tests for grow_connections
# ============================================================================

class TestSTDPUpdate:
    """Tests for DirectionalRangeCompositeMapping.stdp_update()."""

    def test_stdp_causal_potentiates(self):
        """Causal timing (pre before post) should increase weights."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set uniform weights
        for row in matrices:
            for m in row:
                if m is not None:
                    m.forward_weights = torch.full_like(m.forward_weights, 0.5)
        
        # Record initial weights
        initial_dense = composite.to_dense().clone()
        
        # Simulate: src neuron spiked at t=1, dst neuron spikes at t=2
        last_spike = torch.zeros(S * N, dtype=torch.long)
        last_spike[0] = 1  # src neuron 0 spiked at t=1
        
        spiked = torch.zeros(S * N, dtype=torch.bool)
        spiked[N] = True  # dst neuron N (slice 1, local 0) spikes now
        last_spike[N] = 2  # Update dst spike time
        
        composite.stdp_update(
            spiked=spiked,
            last_spike_time=last_spike,
            current_time=2,
            a_pos=1.0, a_neg=0.75, tau_pos=1.0, tau_neg=1.0, eta=0.1,
        )
        
        final_dense = composite.to_dense()
        
        # Connection from 0 -> N should have increased (causal, dt=1 > 0)
        if initial_dense[N, 0] > 0:  # Only if connection exists
            assert final_dense[N, 0] > initial_dense[N, 0], \
                f"Weight should increase: {initial_dense[N, 0]} -> {final_dense[N, 0]}"

    def test_stdp_anticausal_depresses(self):
        """Anti-causal timing (post before pre) should decrease weights."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set uniform weights
        for row in matrices:
            for m in row:
                if m is not None:
                    m.forward_weights = torch.full_like(m.forward_weights, 0.5)
        
        # Record initial weights
        initial_dense = composite.to_dense().clone()
        
        # Simulate: dst neuron spiked at t=1, src neuron spikes at t=2
        last_spike = torch.zeros(S * N, dtype=torch.long)
        last_spike[N] = 1  # dst neuron N spiked at t=1
        
        spiked = torch.zeros(S * N, dtype=torch.bool)
        spiked[0] = True  # src neuron 0 spikes now
        last_spike[0] = 2  # Update src spike time
        
        composite.stdp_update(
            spiked=spiked,
            last_spike_time=last_spike,
            current_time=2,
            a_pos=1.0, a_neg=0.75, tau_pos=1.0, tau_neg=1.0, eta=0.1,
        )
        
        final_dense = composite.to_dense()
        
        # Connection from 0 -> N should have decreased (anti-causal, dt=-1 < 0)
        if initial_dense[N, 0] > 0:  # Only if connection exists
            assert final_dense[N, 0] < initial_dense[N, 0], \
                f"Weight should decrease: {initial_dense[N, 0]} -> {final_dense[N, 0]}"

    def test_stdp_no_spike_no_change(self):
        """No spikes should leave weights unchanged."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set uniform weights
        for row in matrices:
            for m in row:
                if m is not None:
                    m.forward_weights = torch.full_like(m.forward_weights, 0.5)
        
        initial_dense = composite.to_dense().clone()
        
        # No spikes
        spiked = torch.zeros(S * N, dtype=torch.bool)
        last_spike = torch.zeros(S * N, dtype=torch.long)
        
        composite.stdp_update(
            spiked=spiked,
            last_spike_time=last_spike,
            current_time=1,
            a_pos=1.0, a_neg=0.75, tau_pos=1.0, tau_neg=1.0, eta=0.1,
        )
        
        final_dense = composite.to_dense()
        
        assert torch.allclose(final_dense, initial_dense), "Weights changed without spikes"

    def test_stdp_weights_clamped(self):
        """Weights should be clamped to [0, 1]."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set weights near 1.0
        for row in matrices:
            for m in row:
                if m is not None:
                    m.forward_weights = torch.full_like(m.forward_weights, 0.99)
        
        # Strong causal spike
        last_spike = torch.zeros(S * N, dtype=torch.long)
        last_spike[0] = 1
        
        spiked = torch.zeros(S * N, dtype=torch.bool)
        spiked[N] = True
        last_spike[N] = 2
        
        composite.stdp_update(
            spiked=spiked,
            last_spike_time=last_spike,
            current_time=2,
            a_pos=10.0, a_neg=0.75, tau_pos=1.0, tau_neg=1.0, eta=1.0,  # Very high
        )
        
        final_dense = composite.to_dense()
        
        assert final_dense.max() <= 1.0, f"Max weight {final_dense.max()} exceeds 1.0"
        assert final_dense.min() >= 0.0, f"Min weight {final_dense.min()} below 0.0"


class TestGrowConnections:
    """Tests for DirectionalRangeCompositeMapping.grow_connections()."""

    def test_grow_regrows_dead_connections(self):
        """grow_connections regrows zero-weight connections."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Zero all weights
        for row in matrices:
            for m in row:
                if m is not None:
                    with torch.no_grad():
                        m._parameters['_forward_weights_param'].zero_()
                        m._update_reverse_weights()
        
        # Grow with 100% rate
        regrown = composite.grow_connections(rate=1.0)
        
        # Should have regrown all masked positions
        assert regrown > 0
        
        # Verify weights are now non-zero
        for row in matrices:
            for m in row:
                if m is not None:
                    nonzero = (m.forward_weights.abs() > 1e-8) & m.mapping.input_mask
                    assert nonzero.sum() > 0

    def test_grow_respects_rate(self):
        """grow_connections with rate=0 should regrow nothing."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Zero all weights
        for row in matrices:
            for m in row:
                if m is not None:
                    with torch.no_grad():
                        m._parameters['_forward_weights_param'].zero_()
                        m._update_reverse_weights()
        
        # Grow with 0% rate
        regrown = composite.grow_connections(rate=0.0)
        
        assert regrown == 0

    def test_grow_only_affects_dead_connections(self):
        """grow_connections should not modify live (non-zero) connections."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Set all weights to 1.0
        for row in matrices:
            for m in row:
                if m is not None:
                    m.forward_weights = torch.ones_like(m.forward_weights)
        
        # Grow with 100% rate (should do nothing since no dead connections)
        regrown = composite.grow_connections(rate=1.0)
        
        assert regrown == 0
        
        # Verify weights still 1.0 (where masked)
        for row in matrices:
            for m in row:
                if m is not None:
                    live = m.forward_weights[m.mapping.input_mask]
                    assert torch.allclose(live, torch.ones_like(live))

    def test_grow_initializes_small_weights(self):
        """Regrown connections should have small initial weights."""
        N, S, D = 4, 3, 1
        matrices = _build_test_matrices(N, S, D)
        composite = DirectionalRangeCompositeMapping(matrices, N, S, D)
        
        # Zero all weights
        for row in matrices:
            for m in row:
                if m is not None:
                    with torch.no_grad():
                        m._parameters['_forward_weights_param'].zero_()
                        m._update_reverse_weights()
        
        # Grow with 100% rate
        composite.grow_connections(rate=1.0, init_weight=0.01)
        
        # Check regrown weights are small (0.005 to 0.015)
        for row in matrices:
            for m in row:
                if m is not None:
                    live = m.forward_weights[m.mapping.input_mask]
                    assert live.min() >= 0.005 - 1e-6
                    assert live.max() <= 0.015 + 1e-6
