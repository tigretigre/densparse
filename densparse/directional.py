"""DirectionalRangeCompositeMapping for cylinder-like structures."""
from typing import List, Optional, Tuple
import torch
from densparse.composite import CompositeMapping
from densparse.matrix import DenSparseMatrix


class DirectionalRangeCompositeMapping(CompositeMapping):
    """Optimized composite mapping for directional range-limited connectivity.

    This class is designed for cylinder-like structures where:
    - Neurons are organized in S slices of N neurons each
    - Connections go from slice s to slices s, s+1, ..., s+D
    - Each connection pattern within a slice is cyclic

    Key optimizations:
    - Packed weight cache: pre-allocated (num_src, N, width) buffers per ds
      column; avoids torch.stack allocation on every forward pass.
    - Fully batched column operations: all src_slices in a column computed in
      one kernel call — no Python loop over slices.
    - Pre-allocated hot-path buffers per column for use by callers.
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
        weight buffers used by forward and normalize.
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

            # Pre-allocate hot-path buffers exposed to callers via col_data
            dst_times_buf = torch.zeros(num_src, self._N, width, dtype=torch.long, device=device)
            effective_mask_buf = torch.zeros(num_src, self._N, width, dtype=torch.bool, device=device)

            self._column_data.append({
                'ds': ds,
                'width': width,
                'num_src': num_src,
                'matrices': matrices,
                'stacked_indices': stacked_indices,
                'stacked_masks': stacked_masks,
                'active_mask': active_mask,
                'device': device,
                'stacked_dist_factors': None,  # filled by _rebuild_stacked_dist_factors
                'dst_times_buffer': dst_times_buf,
                'effective_mask_buffer': effective_mask_buf,
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

    def _rebuild_stacked_dist_factors(self) -> None:
        """Pre-stack distance factors per column to avoid hot-path allocation.

        Called by set_distance_factors() and to() after distance factors are set/moved.
        Stores a (num_src, N, width) tensor in each col_data['stacked_dist_factors'].
        """
        if self._distance_factors is None:
            for col_data in self._column_data:
                if col_data is not None:
                    col_data['stacked_dist_factors'] = None
            return
        for ds, col_data in enumerate(self._column_data):
            if col_data is None:
                continue
            N = self._N
            width = col_data['width']
            num_src = col_data['num_src']
            device = col_data['device']
            dist_list = [
                self._distance_factors[s][ds]
                if (s < len(self._distance_factors) and
                    ds < len(self._distance_factors[s]) and
                    self._distance_factors[s][ds] is not None)
                else torch.zeros(N, width, device=device)
                for s in range(num_src)
            ]
            col_data['stacked_dist_factors'] = torch.stack(dist_list)

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
        self._rebuild_stacked_dist_factors()

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
        self._rebuild_stacked_dist_factors()
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

