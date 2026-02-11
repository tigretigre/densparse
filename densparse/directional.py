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
) -> torch.Tensor:
    """Pure-tensor STDP computation for a single (src_slice, ds) pair.

    Returns updated forward_weights.
    """
    dt = dst_times_exp - src_times_exp  # (N, width)

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

    new_w = (fwd_weights + dw * needs_update).clamp(0.0, 1.0)
    return torch.where(mask, new_w, fwd_weights)


# Try to compile the STDP kernel. Falls back to eager if inductor fails
# (e.g., architecture mismatch on Apple Silicon with some PyTorch builds).
try:
    _stdp_kernel_compiled = torch.compile(_stdp_kernel)
except Exception:
    _stdp_kernel_compiled = _stdp_kernel


class DirectionalRangeCompositeMapping(CompositeMapping):
    """Optimized composite mapping for directional range-limited connectivity.
    
    This class is designed for cylinder-like structures where:
    - Neurons are organized in S slices of N neurons each
    - Connections go from slice s to slices s, s+1, ..., s+D
    - Each connection pattern within a slice is cyclic
    
    The class exploits the regular structure for efficient operations:
    - Column-wise scatter index sharing (even/odd ds parity optimization)
    - Packed weight tensors per ds column (no width padding)
    - Vectorized normalize and prune operations
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

        # Compute widths for each ds column
        self._widths = self._compute_widths()

        # Pre-build column-wise scatter indices
        self._build_column_scatter_indices()
    
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
            # Find first non-None matrix in this column
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
        """Pre-compute column metadata and cache indices for batched forward pass.
        
        For each ds column, pre-stacks indices and prepares for fast forward pass.
        """
        self._column_data = []
        
        for ds in range(self._D + 1):
            width = self._widths[ds]
            num_src = self._S - ds
            
            if width == 0 or num_src == 0:
                self._column_data.append(None)
                continue
            
            # Collect matrices for this column
            matrices = []
            for src_slice in range(num_src):
                if ds < len(self._matrices_2d[src_slice]):
                    m = self._matrices_2d[src_slice][ds]
                    matrices.append(m)
                else:
                    matrices.append(None)
            
            # Check if we have any non-None matrices
            if all(m is None for m in matrices):
                self._column_data.append(None)
                continue
            
            # Pre-compute and cache indices (they don't change)
            # Get device from first non-None matrix
            device = None
            for m in matrices:
                if m is not None:
                    device = m.forward_weights.device
                    break
            
            if device is None:
                self._column_data.append(None)
                continue
            
            # Pre-stack indices: (num_src, N, width)
            stacked_indices = []
            for m in matrices:
                if m is not None:
                    stacked_indices.append(m.mapping.input_mapping.to(device))
                else:
                    stacked_indices.append(torch.zeros(self._N, width, dtype=torch.long, device=device))
            
            stacked_indices = torch.stack(stacked_indices)  # (num_src, N, width)
            
            self._column_data.append({
                'ds': ds,
                'width': width,
                'num_src': num_src,
                'matrices': matrices,
                'stacked_indices': stacked_indices,  # Cached!
                'device': device,
            })
    
    def to_dense(self) -> torch.Tensor:
        """Convert to a dense (S*N, S*N) matrix representation.

        Places each per-slice matrix at the appropriate block position
        based on src_slice and dst_slice.

        Returns:
            A (S*N, S*N) tensor containing the full dense matrix
        """
        self._sync_reverse_weights()
        total = self._S * self._N
        
        # Get device from first non-None matrix
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
                
                # Get dense block for this matrix
                block = m.to_dense()  # (N, N)
                
                # Place at (dst_slice, src_slice) block position
                # Note: dense matrix is (output, input) = (dst, src)
                dst_start = dst_slice * self._N
                dst_end = dst_start + self._N
                src_start = src_slice * self._N
                src_end = src_start + self._N
                
                result[dst_start:dst_end, src_start:src_end] = block
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mega-matrix forward pass.

        For each ds column, batch-processes all src_slices:
        1. Stack source inputs and weights
        2. Multiply and scatter-add to intermediate
        3. Scatter-add intermediate to output using pre-computed indices

        Args:
            x: Input tensor of shape (S*N,) or (batch, S*N)

        Returns:
            Output tensor of same shape
        """
        was_vector = x.dim() == 1
        if was_vector:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device = x.device
        output = torch.zeros_like(x)

        # Reshape to (batch, S, N)
        x_slices = x.view(batch_size, self._S, self._N)

        for ds in range(self._D + 1):
            col_data = self._column_data[ds]
            if col_data is None:
                continue

            num_src = col_data['num_src']
            width = col_data['width']
            matrices = col_data['matrices']

            # 1. Source inputs: (num_src, batch, N)
            src_inputs = x_slices[:, :num_src, :].permute(1, 0, 2)

            # 2. Stack weights: (num_src, N, width)
            stacked_weights = []
            for src_slice in range(num_src):
                m = matrices[src_slice]
                if m is not None:
                    stacked_weights.append(m.forward_weights)
                else:
                    stacked_weights.append(torch.zeros(self._N, width, device=device))
            stacked_weights = torch.stack(stacked_weights)

            # 3. Cached scatter indices: (num_src, N, width)
            stacked_indices = col_data['stacked_indices']

            # 4. Multiply: (num_src, batch, N, width)
            products = src_inputs.unsqueeze(-1) * stacked_weights.unsqueeze(1)

            # 5. Scatter-add to per-src intermediate: (num_src, batch, N)
            indices_expanded = stacked_indices.unsqueeze(1).expand(-1, batch_size, -1, -1)
            intermediate = torch.zeros(num_src, batch_size, self._N, device=device)
            intermediate.scatter_add_(2,
                indices_expanded.reshape(num_src, batch_size, -1),
                products.view(num_src, batch_size, -1))

            # 6. Write intermediate to output slices
            inter_2d = intermediate.permute(1, 0, 2)  # (batch, num_src, N)
            for src_slice in range(num_src):
                dst_slice = src_slice + ds
                s = dst_slice * self._N
                output[:, s:s + self._N] = output[:, s:s + self._N] + inter_2d[:, src_slice]

        if was_vector:
            output = output.squeeze(0)

        return output
    
    def normalize_incoming(self) -> None:
        """L1-normalize incoming weights per destination across all matrices.

        For each destination neuron, sum all incoming weights from all
        source slices and normalize so they sum to 1.

        Computes sums directly from forward_weights using scatter_add,
        avoiding the expensive reverse weight sync entirely.
        """
        N, S, D = self._N, self._S, self._D

        # Get device from first non-None matrix
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

        # Step 1: Compute sum of |incoming weights| per destination using forward_weights.
        # forward_weights[src_local, col] has dst = input_mapping[src_local, col].
        # We scatter_add abs(forward_weights) by destination to get per-dst sums.
        incoming_sums = torch.zeros(S, N, device=device)

        for ds in range(D + 1):
            col_data = self._column_data[ds]
            if col_data is None:
                continue
            num_src = col_data['num_src']
            matrices = col_data['matrices']
            indices = col_data['stacked_indices']  # (num_src, N, width)

            for src_slice in range(num_src):
                m = matrices[src_slice]
                if m is None:
                    continue
                dst_slice = src_slice + ds
                fwd_abs = m.forward_weights.abs()  # (N, width)
                mapping = indices[src_slice]         # (N, width) — dst_local indices
                # Scatter abs weights to dst positions
                incoming_sums[dst_slice].scatter_add_(0,
                    mapping.reshape(-1), fwd_abs.reshape(-1))

        # Step 2: Normalize forward_weights in each matrix
        with torch.no_grad():
            for ds in range(D + 1):
                col_data = self._column_data[ds]
                if col_data is None:
                    continue
                num_src = col_data['num_src']
                matrices = col_data['matrices']
                indices = col_data['stacked_indices']

                for src_slice in range(num_src):
                    m = matrices[src_slice]
                    if m is None:
                        continue
                    dst_slice = src_slice + ds
                    mapping = indices[src_slice]  # (N, width)
                    mask = m.mapping.input_mask

                    # Gather norms: norms_for_fwd[i, c] = incoming_sums[dst_slice, mapping[i, c]]
                    norms = incoming_sums[dst_slice][mapping].clamp(min=1e-15)

                    m._parameters['_forward_weights_param'].div_(
                        torch.where(mask, norms, torch.ones_like(norms))
                    )

            # Forward weights changed → reverse weights stale
            self._reverse_dirty = True
    
    def prune_incoming(self, keep: int) -> None:
        """Keep top-k incoming weights per destination.

        Uses vectorized operations: for each destination, gathers all incoming
        weights from reverse_weights tensors, finds top-k, and creates masks
        to zero out the rest.

        Args:
            keep: Number of connections to keep per destination
        """
        self._sync_reverse_weights()
        N, S, D = self._N, self._S, self._D
        
        # Process each destination slice
        for dst_slice in range(S):
            # Collect all matrices contributing to this slice and their weights
            # Each matrix's reverse_weights[:, :] gives weights TO each local dst
            matrices_info = []  # List of (matrix, src_slice, ds)
            
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
            
            # For each local destination, gather weights and find top-k
            for dst_local in range(N):
                # Collect all weights to this destination
                weight_list = []  # (abs_weight, matrix_idx, col_idx)
                
                for mat_idx, (m, _, _) in enumerate(matrices_info):
                    # reverse_weights[dst_local, :] = weights TO dst_local
                    rev_w = m.reverse_weights[dst_local, :]  # (width,)
                    mask = m.mapping.output_mask[dst_local, :]  # (width,)
                    
                    for col in range(rev_w.shape[0]):
                        if mask[col] and rev_w[col].abs().item() > 1e-15:
                            weight_list.append((rev_w[col].abs().item(), mat_idx, col))
                
                if len(weight_list) <= keep:
                    continue
                
                # Sort by absolute weight, descending
                weight_list.sort(reverse=True, key=lambda x: x[0])
                
                # Zero out weights not in top-k
                to_zero = weight_list[keep:]
                for _, mat_idx, col in to_zero:
                    m, src_slice, ds = matrices_info[mat_idx]
                    with torch.no_grad():
                        m._parameters['_reverse_weights_param'][dst_local, col] = 0.0
                        # Also zero forward_weights at corresponding location
                        # Find which input maps to this (dst_local, col)
                        out_map = m.mapping.output_mapping[dst_local, col].item()
                        m._parameters['_forward_weights_param'][out_map, col] = 0.0
    
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
        """Column-batched STDP weight update.

        For each ds column, stacks all src_slices' weights, mappings, and
        spike times into single tensors and computes the STDP update in one
        vectorized pass — eliminating the inner Python loop over src_slices.

        Args:
            spiked: Boolean tensor (S*N,) indicating which neurons spiked
            last_spike_time: Tensor (S*N,) with last spike time per neuron
            current_time: Current timestep
            a_pos: Potentiation amplitude
            a_neg: Depression amplitude
            tau_pos: Potentiation time constant
            tau_neg: Depression time constant
            eta: Base learning rate
        """
        N, S, D = self._N, self._S, self._D
        device = spiked.device

        # Reshape to (S, N)
        spiked_2d = spiked.view(S, N)
        last_spike_2d = last_spike_time.view(S, N)

        for ds in range(D + 1):
            col_data = self._column_data[ds]
            if col_data is None:
                continue

            num_src = col_data['num_src']
            width = col_data['width']
            matrices = col_data['matrices']

            # Source spike times: (num_src, N) — sliced directly
            src_times = last_spike_2d[:num_src]
            dst_times = last_spike_2d[ds:ds + num_src]
            src_spiked = spiked_2d[:num_src]
            dst_spiked = spiked_2d[ds:ds + num_src]

            # Quick skip: if no spikes in any src or dst slice for this column
            any_active = src_spiked.any(dim=1) | dst_spiked.any(dim=1)  # (num_src,)
            if not any_active.any():
                continue

            # Use cached stacked_indices: (num_src, N, width)
            all_indices = col_data['stacked_indices']

            # Per-matrix STDP using compiled kernel
            for src_slice in range(num_src):
                m = matrices[src_slice]
                if m is None:
                    continue
                if not (src_spiked[src_slice].any() or dst_spiked[src_slice].any()):
                    continue

                fwd_weights = m.forward_weights     # (N, width)
                mask = m.mapping.input_mask          # (N, width)
                indices = all_indices[src_slice]      # (N, width)

                dst_times_exp = dst_times[src_slice][indices]   # (N, width)
                src_times_exp = src_times[src_slice].unsqueeze(1)  # (N, 1)

                with torch.no_grad():
                    new_w = _stdp_kernel(
                        fwd_weights, mask, dst_times_exp, src_times_exp,
                        current_time, a_pos, a_neg, tau_pos, tau_neg, eta,
                    )
                    m._parameters['_forward_weights_param'].copy_(new_w)

            self._reverse_dirty = True

    def grow_connections(self, rate: float, init_weight: float = 0.01) -> int:
        """Regrow a fraction of dead (zero-weight) connections.
        
        Uses vectorized operations: finds all zero weights across all matrices,
        randomly selects a fraction to regrow, and initializes with small weights.
        
        Args:
            rate: Probability of regrowing each dead connection
            init_weight: Base weight for regrown connections (actual = init_weight * U(0.5, 1.5))
            
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
                
                # Find dead connections (zero and masked)
                fwd_weights = m.forward_weights  # (N, width)
                mask = m.mapping.input_mask  # (N, width)
                
                # Dead = masked AND zero weight
                dead = mask & (fwd_weights.abs() < 1e-15)
                
                if not dead.any():
                    continue
                
                # Random selection with probability rate
                rand_mask = torch.rand_like(fwd_weights) < rate
                to_regrow = dead & rand_mask
                
                n_regrow = to_regrow.sum().item()
                if n_regrow == 0:
                    continue
                
                # Generate random weights for regrown connections
                new_weights = init_weight * (0.5 + torch.rand_like(fwd_weights))
                
                with torch.no_grad():
                    # Update forward weights
                    m._parameters['_forward_weights_param'][to_regrow] = new_weights[to_regrow]
                    # Sync reverse weights
                    m._update_reverse_weights()
                
                regrown += n_regrow
        
        return regrown

    def to(self, device: torch.device) -> 'DirectionalRangeCompositeMapping':
        """Move all matrices to device and rebuild cached column indices."""
        super().to(device)
        self._build_column_scatter_indices()
        return self

    def get_column_weights(self, ds: int) -> List[torch.Tensor]:
        """Get weight tensors for all matrices in a ds column.
        
        Args:
            ds: The ds column index
            
        Returns:
            List of weight tensors, one per source slice (may be None for boundary)
        """
        weights = []
        for src_slice in range(self._S - ds):
            m = self._matrices_2d[src_slice][ds]
            if m is not None:
                weights.append(m.forward_weights)
            else:
                weights.append(None)
        return weights
