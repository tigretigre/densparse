from collections import defaultdict
from typing import Callable, Tuple, List

import networkx as nx
import torch


def bipartite_edge_coloring(row_adj: List[torch.Tensor], col_adj: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Color edges in a bipartite graph to create mapping matrices.

    Args:
        row_adj: List of tensors; row_adj[i] contains column indices j where mask[i,j] == 1.
        col_adj: List of tensors; col_adj[j] contains row indices i where mask[i,j] == 1.

    Returns:
        A tuple of:
          - output_mapping: (output_size, width) tensor mapping outputs to inputs,
          - input_mapping: (input_size, width) tensor mapping inputs to outputs,
        where width is the minimum number of colors needed.
    """
    # Determine dimensions and width
    output_size = len(row_adj)
    input_size = len(col_adj)
    max_out = max(adj.numel() for adj in row_adj) if output_size > 0 else 0
    max_in = max(adj.numel() for adj in col_adj) if input_size > 0 else 0
    width = max(max_out, max_in)
    
    # Initialize mapping matrices with -1 (unassigned)
    output_mapping = torch.full((output_size, width), -1, dtype=torch.long)
    input_mapping = torch.full((input_size, width), -1, dtype=torch.long)
    
    # Build list of edges
    edges = []
    for i, adj in enumerate(row_adj):
        for j in adj.tolist():
            edges.append((i, j))
    uncolored = set(edges)
    
    # --- First pass: assign colors via maximum matchings ---
    for color in range(width):
        H = nx.Graph()
        H.add_nodes_from(f"out{i}" for i in range(output_size))
        H.add_nodes_from(f"in{j}" for j in range(input_size))
        for (i, j) in uncolored:
            H.add_edge(f"out{i}", f"in{j}")
            
        if H.number_of_edges() == 0:
            break
            
        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
            H, top_nodes={f"out{i}" for i in range(output_size)}
        )
        
        for node, partner in matching.items():
            if node.startswith("out"):
                i = int(node[3:])
                j = int(partner[2:])
                if (i, j) in uncolored:
                    output_mapping[i, color] = j
                    input_mapping[j, color] = i
                    uncolored.remove((i, j))
        if not uncolored:
            break
    
    def find_alternating_path(start: Tuple[str, int], a: int, b: int):
        """Follow alternating path using mapping matrices directly."""
        input_indices = []
        output_indices = []
        visited = set()
        current_type, current_idx = start
        
        while True:
            if current_type == 'input':
                input_indices.append(current_idx)
                if input_mapping[current_idx, a] == -1:
                    return input_indices, output_indices
                next_idx = input_mapping[current_idx, a].item()
                current_type, current_idx = 'output', next_idx
            else:  # current_type == 'output'
                output_indices.append(current_idx)
                if output_mapping[current_idx, b] == -1:
                    return input_indices, output_indices
                next_idx = output_mapping[current_idx, b].item()
                current_type, current_idx = 'input', next_idx
                
            if (current_type, current_idx) in visited:
                return None
            visited.add((current_type, current_idx))
    
    def swap_colors_along_path(path_indices, a: int, b: int):
        """Swap colors a and b for all vertices in the path."""
        input_indices, output_indices = path_indices
        if input_indices:
            temp = input_mapping[input_indices, a].clone()
            input_mapping[input_indices, a] = input_mapping[input_indices, b]
            input_mapping[input_indices, b] = temp
        if output_indices:
            temp = output_mapping[output_indices, a].clone()
            output_mapping[output_indices, a] = output_mapping[output_indices, b]
            output_mapping[output_indices, b] = temp
    
    def try_color_edge(i: int, j: int) -> bool:
        """Try to color edge (i,j) using mapping matrices."""
        output_free = output_mapping[i] == -1
        input_free = input_mapping[j] == -1
        common_free = torch.where(output_free & input_free)[0]
        
        if len(common_free) > 0:
            c = common_free[0].item()
            output_mapping[i, c] = j
            input_mapping[j, c] = i
            return True
            
        output_free_idx = torch.where(output_free)[0]
        input_free_idx = torch.where(input_free)[0]
        
        if len(output_free_idx) == 0 or len(input_free_idx) == 0:
            return False
            
        a = output_free_idx[0].item()
        b = input_free_idx[0].item()
        
        path = find_alternating_path(('input', j), a, b)
        if path is None:
            return False
            
        swap_colors_along_path(path, a, b)
        output_mapping[i, a] = j
        input_mapping[j, a] = i
        return True
    
    # --- Second pass: process remaining edges ---
    while uncolored:
        progress = False
        for edge in list(uncolored):
            i, j = edge
            if try_color_edge(i, j):
                uncolored.remove(edge)
                progress = True
        if not progress:
            raise RuntimeError(f"Recoloring failed to color remaining edges: {uncolored}")
    
    return output_mapping, input_mapping


class DenSparseMapping:
    def __init__(
        self,
        input_mapping: torch.Tensor,
        input_mask: torch.Tensor,
        output_mapping: torch.Tensor,
        output_mask: torch.Tensor,
    ):
        """Initialize with pre-computed mapping matrices.

        Args:
            input_mapping: (input_size, width) tensor of output indices
            input_mask: (input_size, width) boolean tensor
            output_mapping: (output_size, width) tensor of input indices
            output_mask: (output_size, width) boolean tensor
        """
        assert input_mapping.shape == input_mask.shape
        assert output_mapping.shape == output_mask.shape
        assert input_mapping.shape[1] == output_mapping.shape[1]

        self.input_mapping = input_mapping
        self.input_mask = input_mask
        self.output_mapping = output_mapping
        self.output_mask = output_mask

        self.input_size = input_mapping.size(0)
        self.output_size = output_mapping.size(0)
        self.mapping_width = input_mapping.size(1)

    @classmethod
    def from_function(
        cls,
        input_size: int,
        output_size: int,
        mapping_width: int,
        mapping_function: Callable[[int, int], Tuple[int, bool]],
    ) -> 'DenSparseMapping':
        """Create mapping matrices from a mapping function.
        
        Raises:
            ValueError: If the mapping function creates invalid bidirectional connections
        """
        max_size = max(input_size, output_size)
        input_mapping = torch.zeros(input_size, mapping_width, dtype=torch.long)
        input_mask = torch.zeros(input_size, mapping_width, dtype=torch.bool)
        output_mapping = torch.zeros(output_size, mapping_width, dtype=torch.long)
        output_mask = torch.zeros(output_size, mapping_width, dtype=torch.bool)

        for row in range(input_size):
            for col in range(mapping_width):
                out_idx, mask = mapping_function(row, col)
                input_mapping[row, col] = out_idx
                input_mask[row, col] = mask
                if mask:
                    if output_mask[out_idx, col]:
                        raise ValueError(f"Multiple inputs ({row} and {output_mapping[out_idx, col]}) mapping to output {out_idx} in column {col}")
                    output_mapping[out_idx, col] = row
                    output_mask[out_idx, col] = mask

        return cls(input_mapping, input_mask, output_mapping, output_mask)

    @classmethod
    def from_mask(cls, mask_matrix: torch.Tensor) -> 'DenSparseMapping':
        """Create mapping matrices from a binary mask matrix.

        Args:
            mask_matrix: (output_size, input_size) binary matrix where 1s
                        indicate allowed connections
        """
        output_size, input_size = mask_matrix.shape
        
        # Create adjacency lists
        row_adj = [torch.nonzero(mask_matrix[i, :]).squeeze(1) for i in range(output_size)]
        col_adj = [torch.nonzero(mask_matrix[:, j]).squeeze(1) for j in range(input_size)]
        
        # Get the colored edge mappings
        output_mapping, input_mapping = bipartite_edge_coloring(row_adj, col_adj)
        
        # Create masks and reset masked entries
        output_mask = output_mapping >= 0
        input_mask = input_mapping >= 0
        output_mapping[~output_mask] = 0
        input_mapping[~input_mask] = 0
        
        return cls(input_mapping, input_mask, output_mapping, output_mask)

    def to_dense(self) -> torch.Tensor:
        """Convert the mapping to a dense binary mask matrix."""
        dense = torch.zeros(self.output_size, self.input_size, dtype=torch.bool)
        for i in range(self.input_size):
            for k in range(self.mapping_width):
                if self.input_mask[i, k]:
                    j = self.input_mapping[i, k]
                    dense[j, i] = True
        return dense

    def transpose(self) -> 'DenSparseMapping':
        """Return a new DenSparseMapping with input and output swapped."""
        return DenSparseMapping(
            input_mapping=self.output_mapping,
            input_mask=self.output_mask,
            output_mapping=self.input_mapping,
            output_mask=self.input_mask,
        )
