import torch
import torch.nn as nn
from typing import Callable, Tuple
from densparse.mapping import DenSparseMapping

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

        # Initialize the forward and reverse weight matrices
        self.forward_weights = nn.Parameter(
            torch.zeros(mapping.input_size, mapping.mapping_width))
        self.reverse_weights = nn.Parameter(
            torch.zeros(mapping.output_size, mapping.mapping_width))

        # Create working area tensor (max_size × mapping_width × max_batch)
        max_size = max(mapping.input_size, mapping.output_size)
        self._working = torch.zeros(max_size, mapping.mapping_width, max_batch)

    @property
    def forward_mapping(self):
        return self.mapping.input_mapping

    @property
    def forward_mask(self):
        return self.mapping.input_mask

    @property
    def reverse_mapping(self):
        return self.mapping.output_mapping

    @property
    def reverse_mask(self):
        return self.mapping.output_mask

    @property
    def input_size(self):
        """Size of input dimension."""
        return self.mapping.input_size

    @property
    def output_size(self):
        """Size of output dimension."""
        return self.mapping.output_size

    def _update_reverse_weights(self):
        """Update reverse weights to match forward weights using the mappings."""
        with torch.no_grad():
            # Zero out working area (using just the first batch slice)
            self._working[self.mapping.input_size:, :, 0].zero_()
            
            # Copy forward weights into working area
            self._working[:self.mapping.input_size, :, 0] = self.forward_weights
            
            # Rearrange each column according to mapping
            for k in range(self.mapping.mapping_width):
                self._working[:self.mapping.output_size, k, 0] = self._working[self.mapping.output_mapping[:, k], k, 0]
            
            # Copy relevant portion to reverse weights
            self.reverse_weights.copy_(self._working[:self.mapping.output_size, :, 0])

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
        # Handle vector vs batch input
        was_vector = x.dim() == 1
        if was_vector:
            # Single input: (input_size,) -> (1, input_size)
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        if batch_size > self.max_batch:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch}")

        # Zero out working area
        self._working[..., :batch_size].zero_()
        
        # Do multiplication with transposed input
        self._working[:self.mapping.input_size, :, :batch_size] = (
            self.forward_weights.unsqueeze(2) * 
            self.forward_mask.unsqueeze(2) * 
            x.t().unsqueeze(1)
        )

        # Rearrange each column according to mapping
        for k in range(self.mapping.mapping_width):
            self._working[:self.output_size, k, :batch_size] = (
                self._working[self.mapping.output_mapping[:, k], k, :batch_size]
            )

        # Apply output mask before summing
        self._working[:self.output_size, :, :batch_size] *= self.reverse_mask.unsqueeze(2)
        
        # Sum and transpose back to (batch_size, output_size)
        result = self._working[:self.mapping.output_size, :, :batch_size].sum(dim=1).t()
        
        # Return vector for single inputs
        return result.squeeze(0) if was_vector else result

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.forward(other)

    def transpose(self) -> 'DenSparseMatrix':
        """Returns a transposed view of this matrix."""
        # Create new matrix with transposed mapping
        transposed = DenSparseMatrix(self.mapping.transpose(), max_batch=self.max_batch)

        # Swap weights
        transposed.forward_weights = self.reverse_weights
        transposed.reverse_weights = self.forward_weights

        return transposed

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
            self.forward_mask[input_idx] & 
            (self.forward_mapping[input_idx] == output_idx)
        )[0]
        reverse_cols = torch.where(
            self.reverse_mask[output_idx] & 
            (self.reverse_mapping[output_idx] == input_idx)
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
            input_mapping: {self.forward_mapping[input_idx]}
            output_mapping: {self.reverse_mapping[output_idx]}
            forward_mask: {self.forward_mask[input_idx]}
            reverse_mask: {self.reverse_mask[output_idx]}
            forward_cols: {forward_cols}
            reverse_cols: {reverse_cols}
            """
        
        with torch.no_grad():
            col = forward_cols[0].item()
            self.forward_weights[input_idx, col] = weight
            self.reverse_weights[output_idx, col] = weight

    def to_dense(self, use_grad: bool = False) -> torch.Tensor:
        """Convert to a dense matrix representation.
        
        Args:
            use_grad: If True, use gradients instead of weights
        
        Returns:
            A (output_size, input_size) tensor containing the dense matrix
        """
        dense = torch.zeros(self.output_size, self.input_size)
        matrix = self.forward_weights.grad if use_grad else self.forward_weights
        
        # For each input, find all its unmasked connections and assign values
        for i in range(self.input_size):
            cols = torch.where(self.forward_mask[i])[0]
            if cols.numel():
                j = self.forward_mapping[i, cols]
                dense[j, i] = matrix[i, cols]
        
        return dense

    def get_grad_matrix(self) -> torch.Tensor:
        """Get gradients as a dense matrix."""
        return self.to_dense(use_grad=True)

    def randomize_weights(self):
        """Initialize weights with random values from N(0,1)."""
        with torch.no_grad():
            self.forward_weights.normal_()
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
            self.forward_mask[input_idx] & 
            (self.forward_mapping[input_idx] == output_idx)
        )[0]
        
        if not cols.numel():
            return 0.0
        
        return self.forward_weights[input_idx, cols[0]].item()

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
            for i in range(matrix.input_size):
                cols = torch.where(matrix.forward_mask[i])[0]
                if cols.numel():
                    j = matrix.forward_mapping[i, cols]
                    matrix.forward_weights[i, cols] = dense[j, i]
            
            matrix._update_reverse_weights()
        
        return matrix

