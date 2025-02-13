import torch
import torch.nn as nn
from densparse.matrix import DenSparseMatrix
from densparse.mapping import DenSparseMapping

class DenSparse(nn.Module):
    def __init__(self, mapping: DenSparseMapping, random_weights: bool = False):
        """Initialize a DenSparse layer.
        
        Args:
            mapping: DenSparseMapping object defining the connectivity pattern
            random_weights: If True, initialize weights randomly instead of zeros
        """
        super().__init__()
        self.matrix = DenSparseMatrix(mapping)
        if random_weights:
            self.matrix.randomize_weights()
        # Register the matrix's weights as our parameter
        self.register_parameter('weights', self.matrix.forward_weights)

    @property
    def input_size(self) -> int:
        return self.matrix.input_size

    @property
    def output_size(self) -> int:
        return self.matrix.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the sparse mapping."""
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size {x.shape[1]} doesn't match expected {self.input_size}")
        
        return self.matrix @ x

    def get_weight(self, from_node: int, to_node: int) -> float:
        """Get weight of connection between two nodes."""
        return self.matrix.get_weight(from_node, to_node)

    def set_weight(self, from_node: int, to_node: int, weight: float):
        """Set weight of connection between two nodes."""
        self.matrix.set_weight(from_node, to_node, weight)

    def to_dense(self) -> torch.Tensor:
        """Convert to dense matrix representation."""
        return self.matrix.to_dense()

    def get_grad_matrix(self) -> torch.Tensor:
        """Get gradients as a dense matrix."""
        return self.matrix.get_grad_matrix()