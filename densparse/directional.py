"""DirectionalRangeCompositeMapping for cylinder-like structures."""
from typing import List, Optional, Tuple
import torch
from densparse.composite import CompositeMapping
from densparse.matrix import DenSparseMatrix


def _stdp_kernel(
    fwd_weights: torch.Tensor,
    mask: torch.Tensor,
    dst_times_exp: torch.Tensor,
    src_times_exp: torch.Tensor,
    current_time: int,
    a_pos: float,
    a_neg: float,
    tau_pos: float,
    tau_neg: float,
    eta: float,
    distance_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-tensor STDP computation. Handles any leading batch dimensions.

    Args:
        fwd_weights:   (..., N, width) — current weights
        mask:          (..., N, width) — valid connections
        dst_times_exp: (..., N, width) — last spike time of dst, expanded to width
        src_times_exp: (..., N, 1) or (..., N, width) — last spike time of src
        current_time:  scalar current timestep
        distance_factor: Optional (..., N, width) per-connection modulation.

    Returns:
        Updated forward_weights of same shape.
    """
    dt = dst_times_exp - src_times_exp  # (..., N, width)

    dst_just_spiked = (dst_times_exp == current_time)
    src_just_spiked = (src_times_exp == current_time)

    needs_update = mask & (
        (dst_just_spiked & (src_times_exp > 0)) |
        (src_just_spiked & (dst_times_exp > 0))
    ) & (dt != 0)

    dt_float = dt.float()
    dw = torch.where(
        dt_float > 0,
        a_pos * (2.0 ** (-dt_float / tau_pos)),
        torch.where(
            dt_float < 0,
            a_neg * (2.0 ** (dt_float / tau_neg)),
            torch.zeros_like(dt_float),
        ),
    ) * dt_float * eta

    if distance_factor is not None:
        dw = dw * distance_factor

    new_w = (fwd_weights + dw * needs_update).clamp(0.0, 1.0)
    return torch.where(mask, new_w, fwd_weights)


# Try to compile the STDP kernel. Falls back to eager if inductor fails.
# Note: torch.compile() itself rarely raises; errors surface on first call.
# We warm-test with a small tensor to surface compilation failures eagerly.
def _try_compile_stdp_kernel():
    try:
        compiled = torch.compile(_stdp_kernel, mode="reduce-overhead")
        # Trigger compilation now; catches broken inductor backends
        _N = 4
        w = torch.zeros(_N, 2)
        m = torch.zeros(_N, 2, dtype=torch.bool)
        t = torch.zeros(_N, 2, dtype=torch.long)
        compiled(w, m, t, t, 1, 1.0, 0.75, 1.0, 1.0, 0.1)
        return compiled
    except Exception:
        return _stdp_kernel

_stdp_kernel_compiled = _try_compile_stdp_kernel()


class DirectionalRangeCompositeMapping(CompositeMapping):
    """Optimized composite mapping for directional range-limited connectivity.

    This class is designed for cylinder-like structures where:
    - Neurons are organized in S slices of N neurons each
    - Connections go from slice s to slices s, s+1, ..., s+D
    - Each connection pattern within a slice is cyclic

    Key optimizations:
    - Packed weight cache: pre-allocated (num_src, N, width) buffers per ds
      column; avoids torch.stack allocation on every forward/STDP call.
    - Fully batched STDP: all src_slices in a column computed in one kernel
      call — no Python loop over slices.
    - compiled _stdp_kernel: uses torch.compile(mode="reduce-overhead").
    - CUDA graph capture helpers for zero-overhead iteration loops.
    """

    def __init__(
        self,
        matrices: List[List[Optional[DenSparseMatrix]]],
        N: int,
        S: int,
        D: int,
    ):
        """Initialize with 2D list of matrices.

        Args:
            matrices: 2D list where matrices[src_slice][ds] is the DenSparseMatrix
                for connections from src_slice to (src_slice + ds), or None if
                that connection doesn't exist (e.g., at boundaries).
            N: Neurons per slice
            S: Number of slices
            D: Maximum distance (ds ranges from 0 to D inclusive)
        """
        # Flatten non-None matrices for base class
        flat = [m for row in matrices for m in row if m is not None]
        super().__init__(flat)

        self._N = N
        self._S = S
        self._D = D
        self._matrices_2d = matrices
        self._reverse_dirty = False
        self._distance_factors = None  # Optional per-matrix distance modulation

        # Packed weight cache: per-column (num_src, N, width) tensor.
        # Rebuilt lazily when _weights_dirty = True.
        self._packed_weights: List[Optional[torch.Tensor]] = []
        self._weights_dirty = True  # Force initial build

        # Compute widths for each ds column
        self._widths = self._compute_widths()

        # Pre-build column-wise scatter indices (also populates _intermediate_buffers)
        self._build_column_scatter_indices()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _sync_reverse_weights(self) -> None:
        """Sync reverse weights from forward weights for all dirty matrices.

        Called lazily before normalize/prune which read reverse_weights.
        """
        if not self._reverse_dirty:
            return
        for row in self._matrices_2d:
            for m in row:
                if m is not None:
                    m._update_reverse_weights()
        self._reverse_dirty = False

    def _compute_widths(self) -> List[int]:
        """Compute mapping width for each ds column."""
        widths = []
        for ds in range(self._D + 1):
            width = 0
            for src_slice in range(self._S):
                if ds < len(self._matrices_2d[src_slice]):
                    m = self._matrices_2d[src_slice][ds]
                    if m is not None:
                        width = m.mapping.mapping_width
                        break
            widths.append(width)
        return widths

    def _build_column_scatter_indices(self):
        """Pre-compute column metadata and cache indices for batched operations.

        For each ds column, pre-stacks indices/masks and allocates packed
        weight buffers used by forward, STDP, and normalize.
        """
        self._column_data = []
        self._packed_weights = []
        self._intermediate_buffers = []

        for ds in range(self._D + 1):
            width = self._widths[ds]
            num_src = self._S - ds

            if width == 0 or num_src == 0:
                self._column_data.append(None)
                self._packed_weights.append(None)
                self._intermediate_buffers.append(None)
                continue

            # Collect matrices for this column
            matrices = []
            for src_slice in range(num_src):
                if ds < len(self._matrices_2d[src_slice]):
                    m = self._matrices_2d[src_slice][ds]
                    matrices.append(m)
                else:
                    matrices.append(None)

            if all(m is None for m in matrices):
                self._column_data.append(None)
                self._packed_weights.append(None)
                self._intermediate_buffers.append(None)
                continue

            # Infer device from first non-None matrix
            device = None
            for m in matrices:
                if m is not None:
                    device = m.forward_weights.device
                    break
            if device is None:
                self._column_data.append(None)
                self._packed_weights.append(None)
                self._intermediate_buffers.append(None)
                continue

            # Pre-stack indices: (num_src, N, width)
            stacked_indices = torch.stack([
                m.mapping.input_mapping.to(device) if m is not None
                else torch.zeros(self._N, width, dtype=torch.long, device=device)
                for m in matrices
            ])

            # Pre-stack masks: (num_src, N, width)
            stacked_masks = torch.stack([
                m.mapping.input_mask.to(device) if m is not None
                else torch.zeros(self._N, width, dtype=torch.bool, device=device)
                for m in matrices
            ])

            # Which src_slices have non-None matrices
            active_mask = torch.tensor(
                [m is not None for m in matrices], dtype=torch.bool, device=device
            )

            # Packed weight buffer: (num_src, N, width) — filled lazily
            packed_w = torch.zeros(num_src, self._N, width, device=device)

            # Pre-allocate intermediate buffer for forward pass (num_src, batch=1, N)
            # This avoids repeated allocations during forward pass
            intermediate_buf = torch.zeros(num_src, 1, self._N, device=device)

            self._column_data.append({
                'ds': ds,
                'width': width,
                'num_src': num_src,
                'matrices': matrices,
                'stacked_indices': stacked_indices,
                'stacked_masks': stacked_masks,
                'active_mask': active_mask,
                'device': device,
            })
            self._packed_weights.append(packed_w)
            self._intermediate_buffers.append(intermediate_buf)

        # Force initial pack
        self._weights_dirty = True

    def _rebuild_packed_weights(self) -> None:
        """Rebuild packed weight buffers from individual matrix parameters.

        Called before forward pass if _weights_dirty is True.
        Each column gets a (num_src, N, width) tensor with stacked weights.
        """
        for ds, col_data in enumerate(self._column_data):
            if col_data is None:
                continue
            pw = self._packed_weights[ds]
            if pw is None:
                continue
            matrices = col_data['matrices']
            width = col_data['width']
            device = col_data['device']
            for s, m in enumerate(matrices):
                if m is not None:
                    pw[s].copy_(m.forward_weights)
                else:
                    pw[s].zero_()
        self._weights_dirty = False

    # ── Forward pass ─────────────────────────────────────────────────────

    def to_dense(self) -> torch.Tensor:
        """Convert to a dense (S*N, S*N) matrix representation."""
        self._sync_reverse_weights()
        total = self._S * self._N

        device = None
        for row in self._matrices_2d:
            for m in row:
                if m is not None:
                    device = m._parameters['_forward_weights_param'].device
                    break
            if device is not None:
                break
        if device is None:
            device = torch.device('cpu')

        result = torch.zeros(total, total, device=device)

        for src_slice in range(self._S):
            for ds in range(len(self._matrices_2d[src_slice])):
                m = self._matrices_2d[src_slice][ds]
                if m is None:
                    continue
                dst_slice = src_slice + ds
                if dst_slice >= self._S:
                    continue
                block = m.to_dense()
                dst_start = dst_slice * self._N
                src_start = src_slice * self._N
                result[dst_start:dst_start + self._N,
                       src_start:src_start + self._N] = block

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batched forward pass using pre-packed weight tensors.

        Rebuilds the packed weight cache if weights were modified since the
        last forward call, then processes all ds columns in a vectorized loop.

        On CUDA, automatically uses torch.autocast(bfloat16) for the multiply-
        scatter operations if bfloat16 is supported (~1.5-2× speedup on
        hardware with bf16 tensor cores, e.g. A100, H100, L4).

        Args:
            x: Input tensor of shape (S*N,) or (batch, S*N)

        Returns:
            Output tensor of same shape (always in input dtype)
        """
        if self._weights_dirty:
            self._rebuild_packed_weights()

        was_vector = x.dim() == 1
        if was_vector:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device = x.device
        output = torch.zeros_like(x)

        # Use bf16 autocast on CUDA for ~1.5-2x throughput gain on modern GPUs
        use_autocast = (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        # Reshape to (batch, S, N)
        x_slices = x.view(batch_size, self._S, self._N)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_autocast):
            for ds in range(self._D + 1):
                col_data = self._column_data[ds]
                if col_data is None:
                    continue

                num_src = col_data['num_src']
                width = col_data['width']

                # (num_src, N, width) — from cache, no allocation
                stacked_weights = self._packed_weights[ds]
                # (num_src, N, width) — pre-computed indices
                stacked_indices = col_data['stacked_indices']

                # Source inputs: (num_src, batch, N)
                src_inputs = x_slices[:, :num_src, :].permute(1, 0, 2)

                # Multiply: (num_src, batch, N, width)
                products = src_inputs.unsqueeze(-1) * stacked_weights.unsqueeze(1)

                # Reuse pre-allocated intermediate buffer for batch_size=1
                # For larger batches, allocate as needed (less common in training loop)
                if batch_size == 1:
                    intermediate = self._intermediate_buffers[ds]
                    intermediate.zero_()
                else:
                    intermediate = torch.zeros(num_src, batch_size, self._N, device=device)

                # Scatter-add to per-src intermediate: (num_src, batch, N)
                indices_expanded = stacked_indices.unsqueeze(1).expand(-1, batch_size, -1, -1)
                intermediate.scatter_add_(2,
                    indices_expanded.reshape(num_src, batch_size, -1),
                    products.view(num_src, batch_size, -1))

                # Write to output: dst slices are [ds, ds+1, ..., ds+num_src-1]
                inter_flat = intermediate.permute(1, 0, 2).reshape(batch_size, -1)
                s = ds * self._N
                e = (ds + num_src) * self._N
                output[:, s:e] = output[:, s:e] + inter_flat

        if was_vector:
            output = output.squeeze(0)

        return output

    # ── STDP ─────────────────────────────────────────────────────────────

    def stdp_update(
        self,
        spiked: torch.Tensor,
        last_spike_time: torch.Tensor,
        current_time: int,
        a_pos: float = 1.0,
        a_neg: float = 0.75,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        eta: float = 0.1,
    ) -> None:
        """Device-adaptive STDP weight update.

        **GPU/accelerator path** (device != CPU):
            All src_slices in each ds column are processed in a single batched
            _stdp_kernel call on (num_src, N, width) tensors. Eliminates
            per-slice Python-dispatched kernel launches — dominant GPU cost.

        **CPU path**:
            Per-slice loop with early spike-skip. On CPU, spike trains are
            sparse (only a few slices active per timestep), so skipping idle
            slices outweighs the Python loop overhead.

        Args:
            spiked:          Boolean tensor (S*N,) — which neurons spiked now
            last_spike_time: Tensor (S*N,) — last spike time per neuron
            current_time:    Current timestep
        """
        N, S, D = self._N, self._S, self._D
        device = spiked.device
        is_gpu = device.type != "cpu"

        spiked_2d = spiked.view(S, N)
        last_spike_2d = last_spike_time.view(S, N)

        with torch.no_grad():
            for ds in range(D + 1):
                col_data = self._column_data[ds]
                if col_data is None:
                    continue

                num_src = col_data['num_src']
                width = col_data['width']
                matrices = col_data['matrices']
                active_mask = col_data['active_mask']  # (num_src,)
                stacked_masks = col_data['stacked_masks']  # (num_src, N, width)
                all_indices = col_data['stacked_indices']  # (num_src, N, width)

                src_spiked = spiked_2d[:num_src]        # (num_src, N)
                dst_spiked = spiked_2d[ds:ds + num_src] # (num_src, N)

                # Quick skip: no spikes anywhere in this column
                any_active = src_spiked.any(dim=1) | dst_spiked.any(dim=1)
                if not any_active.any():
                    continue

                if is_gpu:
                    # ── GPU path: fully batched across all src_slices ────
                    # Single kernel call over (num_src, N, width).
                    # Inactive slices are masked out rather than skipped.

                    # src_times_batch: (num_src, N, 1) → broadcasts
                    src_times_batch = last_spike_2d[:num_src].unsqueeze(2)

                    # dst_times_batch[s, i, c] = last_spike_2d[s+ds, indices[s, i, c]]
                    dst_slice_times = last_spike_2d[ds:ds + num_src]
                    dst_times_batch = torch.gather(
                        dst_slice_times.unsqueeze(2).expand(-1, -1, width),
                        1,
                        all_indices,
                    )  # (num_src, N, width)

                    # Mask out inactive slices
                    slice_active = any_active.view(num_src, 1, 1)
                    effective_mask = stacked_masks & slice_active

                    # Distance modulation
                    dist_factors = None
                    if self._distance_factors is not None:
                        dist_list = [
                            self._distance_factors[s][ds]
                            if (s < len(self._distance_factors) and
                                ds < len(self._distance_factors[s]) and
                                self._distance_factors[s][ds] is not None)
                            else torch.zeros(N, width, device=device)
                            for s in range(num_src)
                        ]
                        dist_factors = torch.stack(dist_list)

                    packed_w = self._packed_weights[ds]
                    new_weights = _stdp_kernel_compiled(
                        packed_w, effective_mask,
                        dst_times_batch, src_times_batch,
                        current_time, a_pos, a_neg, tau_pos, tau_neg, eta,
                        distance_factor=dist_factors,
                    )  # (num_src, N, width)

                    # Write back and sync packed cache
                    for s, m in enumerate(matrices):
                        if m is not None and active_mask[s]:
                            m._parameters['_forward_weights_param'].copy_(new_weights[s])
                    packed_w.copy_(new_weights)

                else:
                    # ── CPU path: per-slice with early spike-skip ────────
                    # Sparse activity means most slices are idle; skip them.
                    for src_slice in range(num_src):
                        if not any_active[src_slice]:
                            continue
                        m = matrices[src_slice]
                        if m is None:
                            continue

                        fwd_weights = m.forward_weights      # (N, width)
                        mask = m.mapping.input_mask           # (N, width)
                        indices = all_indices[src_slice]      # (N, width)

                        dst_times_exp = last_spike_2d[src_slice + ds][indices]
                        src_times_exp = last_spike_2d[src_slice].unsqueeze(1)

                        dist_factor = None
                        if self._distance_factors is not None:
                            if (src_slice < len(self._distance_factors) and
                                    ds < len(self._distance_factors[src_slice])):
                                dist_factor = self._distance_factors[src_slice][ds]

                        new_w = _stdp_kernel_compiled(
                            fwd_weights, mask, dst_times_exp, src_times_exp,
                            current_time, a_pos, a_neg, tau_pos, tau_neg, eta,
                            distance_factor=dist_factor,
                        )
                        m._parameters['_forward_weights_param'].copy_(new_w)

                        # Keep packed cache in sync for this slice
                        pw = self._packed_weights[ds]
                        if pw is not None:
                            pw[src_slice].copy_(new_w)

        # Packed weights are up to date; clean for forward
        self._weights_dirty = False
        # Reverse weights are stale; sync lazily before normalize/prune
        self._reverse_dirty = True

    # ── Weight management ─────────────────────────────────────────────────

    def normalize_incoming(self) -> None:
        """L1-normalize incoming weights per destination across all matrices.

        Vectorized: accumulates sums via scatter_add on packed weight cache,
        then divides in-place. No Python loop over individual neurons.
        """
        N, S, D = self._N, self._S, self._D

        device = None
        for row in self._matrices_2d:
            for m in row:
                if m is not None:
                    device = m.forward_weights.device
                    break
            if device is not None:
                break
        if device is None:
            return

        # Ensure packed weights are current
        if self._weights_dirty:
            self._rebuild_packed_weights()

        # Step 1: compute sum of |incoming weights| per dst neuron
        incoming_sums = torch.zeros(S, N, device=device)

        for ds in range(D + 1):
            col_data = self._column_data[ds]
            if col_data is None:
                continue
            num_src = col_data['num_src']
            indices = col_data['stacked_indices']  # (num_src, N, width)
            width = col_data['width']

            stacked_weights = self._packed_weights[ds]  # (num_src, N, width)
            fwd_abs = stacked_weights.abs()

            flat_abs = fwd_abs.reshape(num_src, -1)        # (num_src, N*width)
            flat_idx = indices.reshape(num_src, -1)        # (num_src, N*width)
            per_slice_sums = torch.zeros(num_src, N, device=device)
            per_slice_sums.scatter_add_(1, flat_idx, flat_abs)
            incoming_sums[ds:ds + num_src] += per_slice_sums

        # Step 2: divide each weight by its destination's incoming sum
        with torch.no_grad():
            for ds in range(D + 1):
                col_data = self._column_data[ds]
                if col_data is None:
                    continue
                num_src = col_data['num_src']
                matrices = col_data['matrices']
                indices = col_data['stacked_indices']  # (num_src, N, width)
                stacked_masks = col_data['stacked_masks']

                dst_sums = incoming_sums[ds:ds + num_src]  # (num_src, N)
                # norms[s, i, c] = incoming_sums[s+ds, indices[s, i, c]]
                norms = torch.gather(
                    dst_sums.unsqueeze(-1).expand(-1, -1, col_data['width']),
                    1,
                    indices,
                ).clamp(min=1e-15)
                divisor = torch.where(stacked_masks, norms, torch.ones_like(norms))

                # Update packed cache in-place
                pw = self._packed_weights[ds]
                pw.div_(divisor)

                # Write back to matrix parameters
                for src_slice, m in enumerate(matrices):
                    if m is not None:
                        m._parameters['_forward_weights_param'].copy_(pw[src_slice])

        # Packed weights just updated; mark clean for forward
        self._weights_dirty = False
        self._reverse_dirty = True

    def prune_incoming(self, keep: int) -> None:
        """Keep top-k incoming weights per destination neuron.

        Vectorized implementation using torch.topk — eliminates the O(S*N)
        Python loop that dominates at large network sizes (54% of total time
        at 128x128 with naive per-neuron Python iteration).

        For each destination slice, gathers all reverse-weight tensors from
        contributing source slices into a single (N, total_incoming) matrix,
        applies topk masking in one call, then scatters the zero-mask back to
        individual matrices' forward and reverse weight parameters.

        Args:
            keep: Number of incoming connections to keep per destination.
        """
        self._sync_reverse_weights()
        N, S, D = self._N, self._S, self._D

        with torch.no_grad():
            for dst_slice in range(S):
                # Collect matrices contributing to this destination slice
                matrices_info = []
                for ds in range(min(D + 1, dst_slice + 1)):
                    src_slice = dst_slice - ds
                    if src_slice < 0:
                        continue
                    if ds >= len(self._matrices_2d[src_slice]):
                        continue
                    m = self._matrices_2d[src_slice][ds]
                    if m is not None:
                        matrices_info.append((m, src_slice, ds))

                if not matrices_info:
                    continue

                # ── Vectorized topk over all incoming connections ─────────
                # Build (N, total_incoming) view of all reverse weights and masks

                rev_chunks = []   # list of (N, w_i) reverse-weight tensors
                mask_chunks = []  # list of (N, w_i) bool mask tensors

                for m, _, _ in matrices_info:
                    rev_chunks.append(m._parameters['_reverse_weights_param'])
                    mask_chunks.append(m.mapping.output_mask)

                all_weights = torch.cat(rev_chunks, dim=1)   # (N, total_incoming)
                all_masks   = torch.cat(mask_chunks, dim=1)  # (N, total_incoming)
                total_incoming = all_weights.shape[1]

                if keep >= total_incoming:
                    continue

                # Which neurons have more than `keep` active incoming connections?
                active_count = (all_masks & (all_weights.abs() > 1e-15)).sum(dim=1)
                has_excess = active_count > keep
                if not has_excess.any():
                    continue

                # Absolute weights for active connections (inactive → 0)
                abs_w = all_weights.abs() * all_masks

                # topk returns top-`keep` indices per row
                actual_k = min(keep, total_incoming)
                _, top_idx = abs_w.topk(actual_k, dim=1, largest=True)

                # Build keep-mask: scatter True at top-k positions
                keep_mask = torch.zeros(N, total_incoming,
                                        dtype=torch.bool, device=abs_w.device)
                keep_mask.scatter_(1, top_idx, True)

                # to_zero: active connections NOT in top-k, for neurons with excess
                to_zero = (all_masks & ~keep_mask) & has_excess.unsqueeze(1)

                if not to_zero.any():
                    continue

                # ── Scatter zeros back to per-matrix parameters ───────────
                offset = 0
                for m, _, _ in matrices_info:
                    w = m._parameters['_reverse_weights_param'].shape[1]
                    z = to_zero[:, offset:offset + w]  # (N, w) slice
                    offset += w

                    if not z.any():
                        continue

                    # Zero reverse weights
                    m._parameters['_reverse_weights_param'][z] = 0.0

                    # Zero corresponding forward weights:
                    # output_mapping[dst_local, col] = src_local for that connection
                    z_rows, z_cols = z.nonzero(as_tuple=True)
                    src_locals = m.mapping.output_mapping[z_rows, z_cols]
                    m._parameters['_forward_weights_param'][src_locals, z_cols] = 0.0

        # Forward weights stale — rebuild packed cache before next forward
        self._weights_dirty = True

    def grow_connections(self, rate: float, init_weight: float = 0.01) -> int:
        """Regrow a fraction of dead (zero-weight) connections.

        Args:
            rate: Probability of regrowing each dead connection
            init_weight: Base weight for regrown connections

        Returns:
            Number of connections regrown
        """
        if rate <= 0:
            return 0

        regrown = 0
        for src_slice in range(self._S):
            for ds in range(min(self._D + 1, len(self._matrices_2d[src_slice]))):
                m = self._matrices_2d[src_slice][ds]
                if m is None:
                    continue

                fwd_weights = m.forward_weights
                mask = m.mapping.input_mask
                dead = mask & (fwd_weights.abs() < 1e-15)

                if not dead.any():
                    continue

                rand_mask = torch.rand_like(fwd_weights) < rate
                to_regrow = dead & rand_mask
                n_regrow = to_regrow.sum().item()
                if n_regrow == 0:
                    continue

                new_weights = init_weight * (0.5 + torch.rand_like(fwd_weights))
                with torch.no_grad():
                    m._parameters['_forward_weights_param'][to_regrow] = new_weights[to_regrow]
                    m._update_reverse_weights()

                regrown += n_regrow

        if regrown > 0:
            self._weights_dirty = True

        return regrown

    def set_distance_factors(
        self,
        factors: List[List[Optional[torch.Tensor]]],
    ) -> None:
        """Set per-connection distance modulation factors.

        Args:
            factors: 2D list matching matrices_2d layout. Each entry is
                either a (N, width) tensor or None.
        """
        self._distance_factors = factors

    def to(self, device: torch.device) -> 'DirectionalRangeCompositeMapping':
        """Move all matrices to device and rebuild cached column indices."""
        super().to(device)
        # Rebuild all cached structures including intermediate buffers
        self._build_column_scatter_indices()
        if self._distance_factors is not None:
            for row in self._distance_factors:
                for i, f in enumerate(row):
                    if f is not None:
                        row[i] = f.to(device)
        return self

    def get_column_weights(self, ds: int) -> List[torch.Tensor]:
        """Get weight tensors for all matrices in a ds column."""
        weights = []
        for src_slice in range(self._S - ds):
            m = self._matrices_2d[src_slice][ds]
            if m is not None:
                weights.append(m.forward_weights)
            else:
                weights.append(None)
        return weights

    # ── CUDA graph helpers ────────────────────────────────────────────────

    def warmup_cuda_graphs(
        self,
        spiked: torch.Tensor,
        last_spike_time: torch.Tensor,
        x_input: torch.Tensor,
        current_time: int = 1,
        a_pos: float = 1.0,
        a_neg: float = 0.75,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        eta: float = 0.1,
        n_warmup: int = 3,
    ) -> None:
        """Run N warm-up iterations so torch.compile can trace and compile.

        Call this once after construction, before the training loop, on
        representative input tensors. Required for CUDA graph capture to work
        correctly and for torch.compile to fully JIT-compile the hot path.

        Args:
            spiked:          (S*N,) bool tensor on target device
            last_spike_time: (S*N,) long tensor on target device
            x_input:         (S*N,) float tensor on target device
            current_time:    Timestep to use for warm-up
            n_warmup:        Number of warm-up forward+STDP iterations
        """
        for _ in range(n_warmup):
            _ = self.forward(x_input)
            self.stdp_update(
                spiked, last_spike_time, current_time,
                a_pos, a_neg, tau_pos, tau_neg, eta,
            )
        if x_input.device.type == "cuda":
            torch.cuda.synchronize()

    def capture_cuda_graph(
        self,
        spiked: torch.Tensor,
        last_spike_time: torch.Tensor,
        x_input: torch.Tensor,
        output: torch.Tensor,
        current_time_tensor: torch.Tensor,
        a_pos: float = 1.0,
        a_neg: float = 0.75,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        eta: float = 0.1,
    ) -> Optional['torch.cuda.CUDAGraph']:
        """Capture the forward+STDP sequence into a CUDA graph.

        CUDA graphs eliminate CPU dispatch overhead, which is the dominant
        cost at small batch sizes (the typical case for this network).

        Expects all tensors to be on CUDA. The caller must:
        1. Call warmup_cuda_graphs() first
        2. Keep spiked / last_spike_time / x_input as static buffers whose
           *data* is updated in-place each timestep (not new allocations).
        3. Replay via graph.replay() each timestep.

        Returns the captured CUDAGraph or None if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return None

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fwd_out = self.forward(x_input)
            output.copy_(fwd_out)
            self.stdp_update(
                spiked, last_spike_time, current_time_tensor.item(),
                a_pos, a_neg, tau_pos, tau_neg, eta,
            )
        return graph
