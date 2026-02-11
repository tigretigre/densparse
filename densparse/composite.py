"""Composite mapping for combining multiple DenSparseMatrix objects."""
from typing import List
import torch
from densparse.matrix import DenSparseMatrix


class CompositeMapping:
    """Generic container for multiple DenSparseMatrix objects.
    
    Provides aggregate operations over a collection of sparse matrices.
    The default forward() sums outputs from all matrices.
    
    Subclasses can override methods to provide optimized implementations
    for specific structures (e.g., DirectionalRangeCompositeMapping).
    """
    
    def __init__(self, matrices: List[DenSparseMatrix]):
        """Initialize with a list of DenSparseMatrix objects.
        
        Args:
            matrices: List of DenSparseMatrix objects. All matrices should
                have compatible input/output sizes for the operations used.
        """
        self.matrices = matrices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sum of outputs from all matrices.
        
        Args:
            x: Input tensor of shape (input_size,) or (batch_size, input_size)
            
        Returns:
            Output tensor of shape (output_size,) or (batch_size, output_size)
        """
        if not self.matrices:
            return torch.zeros_like(x)
        
        result = self.matrices[0].forward(x)
        for m in self.matrices[1:]:
            result = result + m.forward(x)
        return result
    
    def normalize_incoming(self) -> None:
        """L1-normalize incoming weights per destination across all matrices.
        
        This base implementation raises NotImplementedError.
        Subclasses should provide optimized implementations.
        """
        raise NotImplementedError(
            "normalize_incoming() must be implemented by subclass. "
            "Base CompositeMapping does not provide a default implementation."
        )
    
    def prune_incoming(self, keep: int) -> None:
        """Keep top-k incoming weights per destination across all matrices.
        
        Args:
            keep: Number of incoming connections to keep per destination
            
        This base implementation raises NotImplementedError.
        Subclasses should provide optimized implementations.
        """
        raise NotImplementedError(
            "prune_incoming() must be implemented by subclass. "
            "Base CompositeMapping does not provide a default implementation."
        )
    
    def to(self, device: torch.device) -> 'CompositeMapping':
        """Move all matrices to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        for m in self.matrices:
            m.to(device)
        return self
    
    def to_dense(self) -> torch.Tensor:
        """Convert to a dense matrix representation.
        
        Returns the sum of all contained matrices as a dense (output_size, input_size)
        tensor. All matrices must have the same input_size and output_size.
        
        Returns:
            A (output_size, input_size) tensor containing the dense matrix
            
        Raises:
            ValueError: If matrices have inconsistent sizes
        """
        if not self.matrices:
            raise ValueError("Cannot convert empty composite to dense")
        
        input_size = self.matrices[0].input_size
        output_size = self.matrices[0].output_size
        
        # Verify all matrices have the same size
        for m in self.matrices[1:]:
            if m.input_size != input_size or m.output_size != output_size:
                raise ValueError(
                    f"Inconsistent matrix sizes: expected ({output_size}, {input_size}), "
                    f"got ({m.output_size}, {m.input_size})"
                )
        
        # Sum all dense matrices
        result = torch.zeros(output_size, input_size, device=self.matrices[0]._parameters['_forward_weights_param'].device)
        for m in self.matrices:
            result += m.to_dense()
        return result
    
    def __len__(self) -> int:
        """Return number of matrices in the composite."""
        return len(self.matrices)
