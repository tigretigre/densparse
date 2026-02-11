"""Common mapping patterns for DenSparse layers."""
from typing import Callable, Tuple
from densparse.mapping import DenSparseMapping

def up_cycle_mapping(
    input_size: int, output_size: int, max_out: int, offset: int = 0
) -> DenSparseMapping:
    """Create a mapping from smaller to larger layer with cyclic connectivity.
    
    Args:
        input_size: Size of input layer
        output_size: Size of output layer (must be multiple of input_size)
        max_out: Maximum outputs per input node
        offset: Phase shift for the cyclic pattern (default 0). The entire
            output pattern is shifted by this amount. Positive values shift
            outputs to higher indices (with wraparound).
        
    Returns:
        DenSparseMapping with cyclic pattern scaled up
        
    Raises:
        ValueError: If output_size is not a multiple of input_size
        ValueError: If max_out (max_in * ratio) is not an integer
    """
    ratio = output_size // input_size
    if output_size != input_size * ratio:
        raise ValueError(f"Output size {output_size} must be multiple of input size {input_size}")
    
    max_in = max_out // ratio
    width = max_out
    
    def mapping_func(input_idx: int, weight_idx: int) -> Tuple[int, bool]:
        base_output = input_idx * ratio
        local_offset = weight_idx - (max_in - 1) // 2 * ratio
        return ((base_output + local_offset + offset) % output_size, True)
    
    return DenSparseMapping.from_function(input_size, output_size, width, mapping_func)

def down_cycle_mapping(input_size: int, output_size: int, max_out: int) -> DenSparseMapping:
    """Create a mapping from larger to smaller layer with cyclic connectivity.
    
    Args:
        input_size: Size of input layer
        output_size: Size of output layer
        max_out: Maximum outputs per input node
        
    Returns:
        DenSparseMapping with cyclic pattern scaled down
        
    Raises:
        ValueError: If input_size is not a multiple of output_size
    """
    ratio = input_size // output_size
    if input_size != output_size * ratio:
        raise ValueError(f"Input size {input_size} must be multiple of output size {output_size}")
        
    if ratio == 1:
        return up_cycle_mapping(input_size, output_size, max_out)
    
    max_in = max_out * ratio
    width = max_in
    
    def mapping_func(input_idx: int, weight_idx: int) -> Tuple[int, bool]:
        # Each input connects to max_in outputs in its local region
        base_output = input_idx // ratio
        column_group = input_idx % ratio
        
        # If weight_idx is in the column group, return the output
        active_group = weight_idx // max_out
        if active_group == column_group:
            offset = weight_idx % max_out - max_out // 2
            return ((base_output + offset) % output_size, True)
        return (
            output_size + input_idx - base_output - (active_group < column_group),
            False,
        )
    
    return DenSparseMapping.from_function(input_size, output_size, width, mapping_func)

# Alias for the common case of same-size layers
square_cycle_mapping = up_cycle_mapping
