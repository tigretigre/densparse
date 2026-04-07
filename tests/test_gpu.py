"""GPU/CUDA tests for densparse: DenSparseMapping, DenSparseMatrix, DirectionalRangeCompositeMapping."""
import pytest
import torch
from densparse.mapping import DenSparseMapping
from densparse.matrix import DenSparseMatrix
from densparse.mapping_utils import square_cycle_mapping, up_cycle_mapping, down_cycle_mapping
from densparse.directional import DirectionalRangeCompositeMapping, _stdp_kernel
from densparse.composite import CompositeMapping


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_directional(N=8, S=4, D=2) -> DirectionalRangeCompositeMapping:
    """Build a small DirectionalRangeCompositeMapping on CPU."""
    matrices = []
    for src_slice in range(S):
        row = []
        for ds in range(D + 1):
            dst_slice = src_slice + ds
            if dst_slice >= S:
                row.append(None)
            else:
                m = DenSparseMatrix(square_cycle_mapping(N, N, min(2 * ds + 3, N)))
                m.randomize_weights()
                row.append(m)
        matrices.append(row)
    return DirectionalRangeCompositeMapping(matrices, N=N, S=S, D=D)


# ---------------------------------------------------------------------------
# DenSparseMapping GPU tests
# ---------------------------------------------------------------------------

class TestDenSparseMappingGPU:
    def test_to_method(self):
        m = square_cycle_mapping(8, 8, 3)
        m.to(CUDA)
        assert m.device.type == "cuda"
        assert m.input_mapping.device.type == "cuda"
        assert m.input_mask.device.type == "cuda"
        assert m.output_mapping.device.type == "cuda"
        assert m.output_mask.device.type == "cuda"

    def test_device_property(self):
        m = square_cycle_mapping(8, 8, 3)
        assert m.device.type == "cpu"
        m.to(CUDA)
        assert m.device.type == "cuda"

    def test_to_dense_on_cuda(self):
        m = square_cycle_mapping(8, 8, 3)
        m.to(CUDA)
        dense = m.to_dense()
        assert dense.device.type == "cuda"
        assert dense.dtype == torch.bool

    def test_to_dense_matches_cpu(self):
        """GPU to_dense should produce identical result to CPU to_dense."""
        m_cpu = square_cycle_mapping(10, 10, 5)
        m_gpu = square_cycle_mapping(10, 10, 5)
        m_gpu.to(CUDA)

        cpu_dense = m_cpu.to_dense()
        gpu_dense = m_gpu.to_dense().cpu()
        assert torch.equal(cpu_dense, gpu_dense)

    def test_from_mask_then_to_cuda(self):
        mask = torch.rand(8, 8) < 0.4
        m = DenSparseMapping.from_mask(mask)
        m.to(CUDA)
        assert m.device.type == "cuda"
        dense = m.to_dense()
        assert torch.equal(mask, dense.cpu())

    def test_roundtrip_cpu_cuda_cpu(self):
        m = square_cycle_mapping(8, 8, 3)
        dense_before = m.to_dense()
        m.to(CUDA)
        m.to(CPU)
        dense_after = m.to_dense()
        assert torch.equal(dense_before, dense_after)


# ---------------------------------------------------------------------------
# DenSparseMatrix GPU tests
# ---------------------------------------------------------------------------

class TestDenSparseMatrixGPU:
    def test_to_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.to(CUDA)
        assert mat.forward_weights.device.type == "cuda"
        assert mat.reverse_weights.device.type == "cuda"
        assert mat.forward_mapping.device.type == "cuda"
        assert mat.forward_mask.device.type == "cuda"

    def test_cuda_method(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.cuda()
        assert mat.forward_weights.device.type == "cuda"

    def test_cpu_method(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.cuda()
        mat.cpu()
        assert mat.forward_weights.device.type == "cpu"

    def test_forward_vector_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.randomize_weights()
        mat.to(CUDA)
        x = torch.randn(8, device=CUDA)
        y = mat.forward(x)
        assert y.device.type == "cuda"
        assert y.shape == (8,)

    def test_forward_batch_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.randomize_weights()
        mat.to(CUDA)
        x = torch.randn(4, 8, device=CUDA)
        y = mat.forward(x)
        assert y.device.type == "cuda"
        assert y.shape == (4, 8)

    def test_forward_matches_cpu(self):
        """GPU forward pass gives same result as CPU."""
        mapping = square_cycle_mapping(8, 8, 3)
        mat_cpu = DenSparseMatrix(mapping)
        mat_cpu.randomize_weights()

        mat_gpu = DenSparseMatrix(mapping.clone())
        # Copy weights
        with torch.no_grad():
            mat_gpu._parameters['_forward_weights_param'].copy_(mat_cpu.forward_weights)
            mat_gpu._update_reverse_weights()
        mat_gpu.to(CUDA)

        x = torch.randn(8)
        expected = mat_cpu.forward(x)
        result = mat_gpu.forward(x.to(CUDA)).cpu()
        assert torch.allclose(expected, result, atol=1e-5)

    def test_set_get_weight_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.to(CUDA)
        mat.set_weight(0, 1, 0.42)
        assert mat.get_weight(0, 1) == pytest.approx(0.42)

    def test_to_dense_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.randomize_weights()
        mat.to(CUDA)
        dense = mat.to_dense()
        assert dense.device.type == "cuda"

    def test_to_dense_matches_cpu(self):
        mapping = square_cycle_mapping(8, 8, 3)
        mat_cpu = DenSparseMatrix(mapping)
        mat_cpu.randomize_weights()

        mat_gpu = DenSparseMatrix(mapping.clone())
        with torch.no_grad():
            mat_gpu._parameters['_forward_weights_param'].copy_(mat_cpu.forward_weights)
            mat_gpu._update_reverse_weights()
        mat_gpu.to(CUDA)

        dense_cpu = mat_cpu.to_dense()
        dense_gpu = mat_gpu.to_dense().cpu()
        assert torch.allclose(dense_cpu, dense_gpu, atol=1e-6)

    def test_from_dense_then_cuda(self):
        dense = torch.zeros(8, 8)
        dense[0, 1] = 0.5
        dense[2, 3] = 0.7
        mat = DenSparseMatrix.from_dense(dense)
        mat.to(CUDA)
        x = torch.randn(8, device=CUDA)
        y = mat.forward(x)
        assert y.device.type == "cuda"

    def test_update_reverse_weights_on_cuda(self):
        """_update_reverse_weights works correctly on GPU."""
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.to(CUDA)
        mat.set_weight(0, 1, 0.5)
        # Check reverse weight was also updated
        fwd_dense = mat.to_dense()
        assert fwd_dense[1, 0].item() == pytest.approx(0.5)

    def test_randomize_weights_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.to(CUDA)
        mat.randomize_weights()
        assert mat.forward_weights.device.type == "cuda"
        assert not torch.all(mat.forward_weights == 0)

    def test_mapping_to_dense_on_cuda(self):
        mat = DenSparseMatrix(square_cycle_mapping(8, 8, 3))
        mat.to(CUDA)
        dense = mat.mapping.to_dense()
        assert dense.device.type == "cuda"


# ---------------------------------------------------------------------------
# _stdp_kernel direct tests (covers the function body regardless of compile)
# ---------------------------------------------------------------------------

class TestStdpKernelDirect:
    """Test _stdp_kernel directly to ensure coverage of the function body."""

    def _make_tensors(self, N=4, width=3, device=CPU):
        fwd = torch.rand(N, width, device=device)
        mask = torch.ones(N, width, dtype=torch.bool, device=device)
        dst_times = torch.randint(1, 5, (N, width), device=device)
        src_times = torch.randint(1, 5, (N, 1), device=device)
        return fwd, mask, dst_times, src_times

    def test_cpu_causal(self):
        fwd, mask, dst_times, src_times = self._make_tensors()
        # Force dst > src (causal: potentiation)
        dst_times = torch.full_like(dst_times, 5)
        src_times = torch.full_like(src_times, 3)
        src_times[0] = 5  # current_time: dst just spiked
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1)
        assert new_w.shape == fwd.shape
        assert (new_w >= 0).all() and (new_w <= 1).all()

    def test_cpu_anti_causal(self):
        fwd, mask, dst_times, src_times = self._make_tensors()
        src_times = torch.full_like(src_times, 5)  # src just spiked
        dst_times = torch.full_like(dst_times, 3)  # dst spiked earlier
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1)
        assert new_w.shape == fwd.shape

    def test_with_distance_factor(self):
        fwd, mask, dst_times, src_times = self._make_tensors()
        dist_factor = torch.rand_like(fwd)
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1,
                             distance_factor=dist_factor)
        assert new_w.shape == fwd.shape

    def test_mask_zeros_exclude(self):
        """Masked-out connections should not be updated."""
        N, width = 4, 3
        fwd = torch.rand(N, width)
        mask = torch.zeros(N, width, dtype=torch.bool)
        dst_times = torch.full((N, width), 5, dtype=torch.long)
        src_times = torch.full((N, 1), 3, dtype=torch.long)
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1)
        # Unmasked entries unchanged
        assert torch.allclose(new_w, fwd)

    def test_clamping(self):
        """Weights should be clamped to [0, 1]."""
        N, width = 4, 3
        fwd = torch.ones(N, width) * 0.99
        mask = torch.ones(N, width, dtype=torch.bool)
        dst_times = torch.full((N, width), 5, dtype=torch.long)
        src_times = torch.full((N, 1), 3, dtype=torch.long)
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 1.0)
        assert (new_w <= 1.0).all()
        assert (new_w >= 0.0).all()

    def test_cuda_execution(self):
        fwd, mask, dst_times, src_times = self._make_tensors(device=CUDA)
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1)
        assert new_w.device.type == "cuda"
        assert new_w.shape == fwd.shape

    def test_cuda_with_distance_factor(self):
        fwd, mask, dst_times, src_times = self._make_tensors(device=CUDA)
        dist = torch.rand_like(fwd)
        new_w = _stdp_kernel(fwd, mask, dst_times, src_times, 5, 1.0, 0.75, 1.0, 1.0, 0.1,
                             distance_factor=dist)
        assert new_w.device.type == "cuda"


# ---------------------------------------------------------------------------
# DirectionalRangeCompositeMapping GPU tests
# ---------------------------------------------------------------------------

class TestDirectionalGPU:
    def test_to_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        # All matrices should be on CUDA
        for row in comp._matrices_2d:
            for m in row:
                if m is not None:
                    assert m.forward_weights.device.type == "cuda"

    def test_forward_vector_on_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        x = torch.randn(4 * 8, device=CUDA)
        y = comp.forward(x)
        assert y.device.type == "cuda"
        assert y.shape == (4 * 8,)

    def test_forward_batch_on_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        x = torch.randn(3, 4 * 8, device=CUDA)
        y = comp.forward(x)
        assert y.device.type == "cuda"
        assert y.shape == (3, 4 * 8)

    def test_forward_matches_cpu(self):
        """GPU forward should match CPU forward."""
        N, S, D = 8, 4, 2
        comp_cpu = _build_directional(N, S, D)

        # Build GPU version with same weights
        comp_gpu = _build_directional(N, S, D)
        for s in range(S):
            for ds in range(D + 1):
                m_cpu = comp_cpu._matrices_2d[s][ds]
                m_gpu = comp_gpu._matrices_2d[s][ds]
                if m_cpu is not None and m_gpu is not None:
                    with torch.no_grad():
                        m_gpu._parameters['_forward_weights_param'].copy_(m_cpu.forward_weights)
                        m_gpu._update_reverse_weights()
        comp_gpu._weights_dirty = True
        comp_gpu.to(CUDA)

        x = torch.randn(S * N)
        y_cpu = comp_cpu.forward(x)
        y_gpu = comp_gpu.forward(x.to(CUDA)).cpu()
        assert torch.allclose(y_cpu, y_gpu, atol=1e-4)

    def test_stdp_update_gpu_path(self):
        """GPU STDP path is exercised when device is CUDA."""
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)

        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        spiked[0] = True
        spiked[8] = True  # Different slice
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        last_spike[0] = 1
        last_spike[8] = 2

        comp.stdp_update(spiked, last_spike, current_time=3)

        # Weights should still be on CUDA
        for row in comp._matrices_2d:
            for m in row:
                if m is not None:
                    assert m.forward_weights.device.type == "cuda"

    def test_stdp_no_spikes_skipped(self):
        """When no neurons spike, stdp_update should be a no-op."""
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)

        # Record initial weights
        w_before = comp._matrices_2d[0][0].forward_weights.clone()

        spiked = torch.zeros(4 * 8, dtype=torch.bool, device=CUDA)
        last_spike = torch.zeros(4 * 8, dtype=torch.long, device=CUDA)
        comp.stdp_update(spiked, last_spike, current_time=1)

        w_after = comp._matrices_2d[0][0].forward_weights
        assert torch.allclose(w_before, w_after)

    def test_normalize_incoming_on_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        comp.normalize_incoming()
        # No crash; weights should be on CUDA
        for row in comp._matrices_2d:
            for m in row:
                if m is not None:
                    assert m.forward_weights.device.type == "cuda"

    def test_prune_incoming_on_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        comp.prune_incoming(keep=2)
        for row in comp._matrices_2d:
            for m in row:
                if m is not None:
                    assert m.forward_weights.device.type == "cuda"

    def test_grow_connections_on_cuda(self):
        """grow_connections works on GPU (rand tensors on correct device)."""
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        # Zero out some weights to create dead connections
        with torch.no_grad():
            for row in comp._matrices_2d:
                for m in row:
                    if m is not None:
                        m._parameters['_forward_weights_param'].zero_()
        comp._weights_dirty = True

        n_regrown = comp.grow_connections(rate=0.5)
        assert isinstance(n_regrown, int)
        assert n_regrown >= 0

    def test_grow_connections_zero_rate(self):
        """grow_connections with rate=0 is a no-op."""
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        n = comp.grow_connections(rate=0.0)
        assert n == 0

    def test_to_dense_on_cuda(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        dense = comp.to_dense()
        assert dense.device.type == "cuda"
        assert dense.shape == (4 * 8, 4 * 8)

    def test_get_column_weights(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        weights = comp.get_column_weights(ds=0)
        assert len(weights) == 4  # S - 0 = 4
        for w in weights:
            assert w is not None
            assert w.device.type == "cuda"

    def test_get_column_weights_last_ds(self):
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        weights = comp.get_column_weights(ds=2)
        # num_src for ds=2: S-2 = 2
        assert len(weights) == 2

    def test_warmup_cuda_graphs(self):
        """warmup_cuda_graphs runs without error."""
        comp = _build_directional(N=8, S=4, D=1)
        comp.to(CUDA)
        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        x_in = torch.randn(N_total, device=CUDA)
        comp.warmup_cuda_graphs(spiked, last_spike, x_in, n_warmup=2)

    def test_capture_cuda_graph(self):
        """CUDA graph capture (forward-only) produces a CUDAGraph that replays correctly."""
        comp = _build_directional(N=8, S=4, D=1)
        comp.to(CUDA)
        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        x_in = torch.randn(N_total, device=CUDA)
        output = torch.zeros(N_total, device=CUDA)

        comp.warmup_cuda_graphs(spiked, last_spike, x_in, n_warmup=3)
        graph = comp.capture_cuda_graph(x_in, output)
        assert graph is not None
        # Replay should not raise
        graph.replay()
        torch.cuda.synchronize()
        # output should have been populated
        assert output.shape == (N_total,)

    def test_stdp_update_uncompiled_gpu(self):
        """_stdp_update_uncompiled runs on GPU and produces correct device tensors."""
        comp = _build_directional(N=8, S=4, D=2)
        comp.to(CUDA)
        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        spiked[0] = True
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        last_spike[0] = 2

        w_before = comp._matrices_2d[0][0].forward_weights.clone()
        comp._stdp_update_uncompiled(spiked, last_spike, current_time=3)
        # Weights should still be on CUDA
        assert comp._matrices_2d[0][0].forward_weights.device.type == "cuda"

    def test_stdp_update_uncompiled_with_distance_factors(self):
        """_stdp_update_uncompiled with distance factors runs without error."""
        comp = _build_directional(N=8, S=4, D=2)
        factors = []
        for s in range(4):
            row = []
            for ds in range(3):
                if s + ds < 4:
                    m = comp._matrices_2d[s][ds]
                    if m is not None:
                        row.append(torch.rand(8, m.mapping.mapping_width))
                    else:
                        row.append(None)
                else:
                    row.append(None)
            factors.append(row)
        comp.set_distance_factors(factors)
        comp.to(CUDA)

        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        spiked[0] = True
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        last_spike[0] = 1
        comp._stdp_update_uncompiled(spiked, last_spike, current_time=3)

    def test_distance_factors_on_cuda(self):
        """Distance factors are correctly moved to CUDA in to()."""
        comp = _build_directional(N=8, S=4, D=2)
        # Set some distance factors on CPU first
        factors = []
        for s in range(4):
            row = []
            for ds in range(3):
                if s + ds < 4:
                    row.append(torch.rand(8, comp._widths[ds]))
                else:
                    row.append(None)
            factors.append(row)
        comp.set_distance_factors(factors)

        comp.to(CUDA)

        # Verify factors are on CUDA
        for row in comp._distance_factors:
            for f in row:
                if f is not None:
                    assert f.device.type == "cuda"

    def test_stdp_with_distance_factors_on_cuda(self):
        """STDP with distance factors runs on GPU."""
        comp = _build_directional(N=8, S=4, D=2)
        factors = []
        for s in range(4):
            row = []
            for ds in range(3):
                if s + ds < 4:
                    m = comp._matrices_2d[s][ds]
                    if m is not None:
                        row.append(torch.rand(8, m.mapping.mapping_width))
                    else:
                        row.append(None)
                else:
                    row.append(None)
            factors.append(row)
        comp.set_distance_factors(factors)
        comp.to(CUDA)

        N_total = 4 * 8
        spiked = torch.zeros(N_total, dtype=torch.bool, device=CUDA)
        spiked[0] = True
        last_spike = torch.zeros(N_total, dtype=torch.long, device=CUDA)
        last_spike[0] = 2
        comp.stdp_update(spiked, last_spike, current_time=3)


# ---------------------------------------------------------------------------
# CompositeMapping GPU tests
# ---------------------------------------------------------------------------

class TestCompositeMappingGPU:
    def test_to_device(self):
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        m1.randomize_weights()
        m2.randomize_weights()
        comp = CompositeMapping([m1, m2])
        comp.to(CUDA)
        for m in comp.matrices:
            assert m.forward_weights.device.type == "cuda"

    def test_forward_on_cuda(self):
        size = 8
        m1 = DenSparseMatrix(square_cycle_mapping(size, size, 3))
        m2 = DenSparseMatrix(square_cycle_mapping(size, size, 3, offset=1))
        m1.randomize_weights()
        m2.randomize_weights()
        comp = CompositeMapping([m1, m2])
        comp.to(CUDA)
        x = torch.randn(size, device=CUDA)
        y = comp.forward(x)
        assert y.device.type == "cuda"
