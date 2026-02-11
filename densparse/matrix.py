import torch
import torch.nn as nn
from typing import Callable, Tuple
from densparse.mapping import DenSparseMapping
import time

INPUT_BATCH_SLICE = 0
OUTPUT_BATCH_SLICE = 1

class DenSparseMatrix(nn.Module):
    def __init__(self, mapping: DenSparseMapping, max_batch: int = 32):
        """
        Args:
            mapping: DenSparseMapping object containing the mapping matrices
            max_batch: Maximum batch size for matrix multiplication
        """
        super().__init__()
        self.mapping = mapping
        self.max_batch = max_batch

        # Initialize the forward and reverse weight matrices as parameters
        self.register_parameter(
            '_forward_weights_param',
            nn.Parameter(torch.zeros(mapping.input_size, mapping.mapping_width))
        )
        self.register_parameter(
            '_reverse_weights_param',
            nn.Parameter(torch.zeros(mapping.output_size, mapping.mapping_width))
        )

        # Create working area tensor as a buffer
        max_size = max(mapping.input_size, mapping.output_size)
        self.register_buffer(
            '_working',
            torch.zeros(max_size, mapping.mapping_width, 2 * max(1, max_batch))
        )

    @property
    def input_size(self) -> int:
        """Size of input dimension."""
        return self.mapping.input_size

    @property
    def output_size(self) -> int:
        """Size of output dimension."""
        return self.mapping.output_size

    @property
    def forward_mask(self) -> torch.Tensor:
        """Input-side mask matrix."""
        return self.mapping.input_mask

    @property
    def reverse_mask(self) -> torch.Tensor:
        """Output-side mask matrix."""
        return self.mapping.output_mask

    @property
    def forward_mapping(self) -> torch.Tensor:
        """Input-side mapping matrix."""
        return self.mapping.input_mapping

    @property
    def reverse_mapping(self) -> torch.Tensor:
        """Output-side mapping matrix."""
        return self.mapping.output_mapping

    @property
    def forward_weights(self) -> torch.Tensor:
        """Input-side weight matrix."""
        return self._parameters['_forward_weights_param']

    @forward_weights.setter
    def forward_weights(self, value: torch.Tensor):
        """Set input-side weights, applying mask and syncing reverse weights."""
        with torch.no_grad():
            self._parameters['_forward_weights_param'].copy_(value)
            self._parameters['_forward_weights_param'] *= self.mapping.input_mask
            self._update_reverse_weights()

    @property
    def reverse_weights(self) -> torch.Tensor:
        """Output-side weight matrix."""
        return self._parameters['_reverse_weights_param']

    def _update_reverse_weights(self):
        """Update reverse weights to match forward weights using the mappings."""
        with torch.no_grad():
            # Zero working area (using just the first two batch slice)
            self._buffers['_working'][self.mapping.input_size:self.mapping.output_size, :, INPUT_BATCH_SLICE:OUTPUT_BATCH_SLICE+1].zero_()
            self._buffers['_working'][:self.mapping.input_size, :, INPUT_BATCH_SLICE].copy_(
                self._parameters['_forward_weights_param']
            )
            # Scatter from input region to output region (using second batch slice)
            self._buffers['_working'][:max(self.mapping.input_size, self.mapping.output_size), :, OUTPUT_BATCH_SLICE].scatter_(
                0,  # dim to scatter along (output dimension)
                self.mapping.input_mapping,  # indices tensor
                self._buffers['_working'][:self.mapping.input_size, :, INPUT_BATCH_SLICE]  # source
            )
            # Copy to reverse weights
            self._parameters['_reverse_weights_param'].copy_(
                self._buffers['_working'][:self.mapping.output_size, :, OUTPUT_BATCH_SLICE]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic handling of vector vs matrix input.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) for batches
               or (input_size,) for single vectors
            
        Returns:
            Output tensor of shape (batch_size, output_size) for batches
            or (output_size,) for single vectors
        """
        return self._multiply_matrix(x)

    def _multiply_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply a matrix by this matrix using the mapping.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) for batches
               or (input_size,) for single inputs
            
        Returns:
            Output tensor of shape (batch_size, output_size) for batches
            or (output_size,) for single inputs
        """
        #t0 = time.perf_counter()
        # Handle vector vs batch input
        was_vector = x.dim() == 1
        if was_vector:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        if batch_size > self.max_batch:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch}")
        #t1 = time.perf_counter()

        # Zero out working area
        self._buffers['_working'][:self.mapping.input_size, :, :batch_size].copy_(
            self._parameters['_forward_weights_param'].unsqueeze(2)
        )
        #t2 = time.perf_counter()
        
        # Do multiplication with transposed input
        self._buffers['_working'][:self.mapping.input_size, :, :batch_size] *= x.t().unsqueeze(1)
        #t3 = time.perf_counter()
        
        # Create index tensor for scatter operation
        index = self.mapping.input_mapping.unsqueeze(-1).expand(-1, -1, batch_size)
        
        # Scatter directly into output region of working buffer
        self._buffers['_working'][:max(self.mapping.input_size, self.mapping.output_size), :, batch_size:2*batch_size].scatter_(
            0,  # dim to scatter along
            index,  # indices tensor
            self._buffers['_working'][:self.mapping.input_size, :, :batch_size]  # source
        )
        
        # Sum along mapping width dimension
        result = self._buffers['_working'][:self.mapping.output_size, :, batch_size:2*batch_size].sum(dim=1)
        
        # Return single vector if input was single vector
        if x.size(0) == 1:
            result = result.squeeze(1)
        else:
            result = result.t()
        
        #t4 = time.perf_counter()

        #print(f"\nProfiling _multiply_matrix:")
        #print(f"Input handling: {(t1-t0)*1000:.3f}ms")
        #print(f"Zero working: {(t2-t1)*1000:.3f}ms")
        #print(f"Matrix multiply: {(t3-t2)*1000:.3f}ms")
        #print(f"Scatter: {(t4-t3)*1000:.3f}ms")
        #print(f"Final reshape: {(t4-t0)*1000:.3f}ms")
        
        return result

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.forward(other)

    def transpose(self) -> 'DenSparseMatrix':
        """Returns a transposed view of this matrix."""
        # Create new matrix with transposed mapping
        transposed = DenSparseMatrix(self.mapping.transpose(), max_batch=self.max_batch)
        
        # Swap weights
        transposed._parameters['_forward_weights_param'] = self._parameters['_reverse_weights_param']
        transposed._parameters['_reverse_weights_param'] = self._parameters['_forward_weights_param']
        
        return transposed

    def to(self, device: torch.device) -> 'DenSparseMatrix':
        """Move matrix to specified device."""
        # Move parameters and buffers
        self._parameters['_forward_weights_param'] = nn.Parameter(self._parameters['_forward_weights_param'].to(device))
        self._parameters['_reverse_weights_param'] = nn.Parameter(self._parameters['_reverse_weights_param'].to(device))
        self._buffers['_working'] = self._buffers['_working'].to(device)
        
        # Move mapping tensors
        self.mapping.input_mapping = self.mapping.input_mapping.to(device)
        self.mapping.input_mask = self.mapping.input_mask.to(device)
        self.mapping.output_mapping = self.mapping.output_mapping.to(device)
        self.mapping.output_mask = self.mapping.output_mask.to(device)
        
        return self

    def set_weight(self, input_idx: int, output_idx: int, weight: float, ignore_unmapped: bool = False) -> None:
        """Set weight for connection between input_idx and output_idx.
        
        Args:
            input_idx: Input index
            output_idx: Output index
            weight: Weight value to set
            ignore_unmapped: If True, silently ignore attempts to set unmapped connections
            
        Raises:
            ValueError: If the connection doesn't exist in the mapping or is masked (when ignore_unmapped=False)
            AssertionError: If there are duplicate connections (should never happen)
        """
        # Find matching columns in forward and reverse mappings
        forward_cols = torch.where(
            self.mapping.input_mask[input_idx] & 
            (self.mapping.input_mapping[input_idx] == output_idx)
        )[0]
        reverse_cols = torch.where(
            self.mapping.output_mask[output_idx] & 
            (self.mapping.output_mapping[output_idx] == input_idx)
        )[0]
        
        if not forward_cols.numel():
            if ignore_unmapped:
                return
            else:
                raise ValueError(f"Connection between input {input_idx} and output {output_idx} does not exist in the mapping")
        
        # Verify exactly one connection exists
        assert forward_cols.numel() == 1 and reverse_cols.numel() == 1, \
            f"""
            Found multiple connections between input {input_idx} and output {output_idx}
            input_mapping: {self.mapping.input_mapping[input_idx]}
            output_mapping: {self.mapping.output_mapping[output_idx]}
            forward_mask: {self.mapping.input_mask[input_idx]}
            reverse_mask: {self.mapping.output_mask[output_idx]}
            forward_cols: {forward_cols}
            reverse_cols: {reverse_cols}
            """
        
        with torch.no_grad():
            col = forward_cols[0].item()
            self._parameters['_forward_weights_param'][input_idx, col] = weight
            self._parameters['_reverse_weights_param'][output_idx, col] = weight

    def to_dense(self, use_grad: bool = False) -> torch.Tensor:
        """Convert to a dense matrix representation.
        
        Args:
            use_grad: If True, use gradients instead of weights
        
        Returns:
            A (output_size, input_size) tensor containing the dense matrix
        """
        matrix = self._parameters['_forward_weights_param'].grad if use_grad else self._parameters['_forward_weights_param']
        dense = torch.zeros(self.mapping.output_size, self.mapping.input_size, device=matrix.device)
        
        # For each input, find all its unmasked connections and assign values
        for i in range(self.mapping.input_size):
            cols = torch.where(self.mapping.input_mask[i])[0]
            if cols.numel():
                j = self.mapping.input_mapping[i, cols]
                dense[j, i] = matrix[i, cols]
        
        return dense

    def get_grad_matrix(self) -> torch.Tensor:
        """Get gradients as a dense matrix."""
        return self.to_dense(use_grad=True)

    def randomize_weights(self):
        """Initialize weights with random values from N(0,1)."""
        with torch.no_grad():
            self._parameters['_forward_weights_param'].normal_()
            self._parameters['_forward_weights_param'] *= self.mapping.input_mask
            self._update_reverse_weights()

    def get_weight(self, input_idx: int, output_idx: int) -> float:
        """Get weight for connection between input_idx and output_idx.
        
        Args:
            input_idx: Input index
            output_idx: Output index
            
        Returns:
            Weight value, or 0.0 if connection doesn't exist
        """
        # Find matching column in forward mapping
        cols = torch.where(
            self.mapping.input_mask[input_idx] & 
            (self.mapping.input_mapping[input_idx] == output_idx)
        )[0]
        
        if not cols.numel():
            return 0.0
        
        return self._parameters['_forward_weights_param'][input_idx, cols[0]].item()

    @classmethod
    def from_dense(cls, dense: torch.Tensor, max_batch: int = 32) -> 'DenSparseMatrix':
        """Create DenSparseMatrix from dense matrix.
        
        Args:
            dense: Dense matrix to convert
            max_batch: Maximum batch size for matrix multiplication
            
        Returns:
            New DenSparseMatrix with same connectivity as dense matrix
        """
        # Create mapping from non-zero pattern
        mask = (dense != 0)
        mapping = DenSparseMapping.from_mask(mask)
        
        # Create matrix and copy weights
        matrix = cls(mapping, max_batch=max_batch)
        
        with torch.no_grad():
            # For each input, find all its unmasked connections
            for i in range(matrix.mapping.input_size):
                cols = torch.where(matrix.mapping.input_mask[i])[0]
                if cols.numel():
                    j = matrix.mapping.input_mapping[i, cols]
                    matrix._parameters['_forward_weights_param'][i, cols] = dense[j, i]
            
            matrix._update_reverse_weights()
        
        return matrix

    def cpu(self) -> 'DenSparseMatrix':
        """Move matrix to CPU."""
        return self.to(torch.device('cpu'))

    def cuda(self) -> 'DenSparseMatrix':
        """Move matrix to CUDA device."""
        return self.to(torch.device('cuda'))

    def mps(self) -> 'DenSparseMatrix':
        """Move matrix to MPS device."""
        return self.to(torch.device('mps'))

