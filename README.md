# DenSparse

A PyTorch-based implementation of sparse neural networks using dense matrix representation with deterministic connectivity patterns.

## Installation

```bash
pip install -e .
```

## Usage

```python
from densparse import DenSparseMatrix

# Create a mapping function
def mapping_func(row_idx: int, col_idx: int) -> tuple[int, int]:
    # Return (output_idx, mask_value) pairs
    ...

# Create a sparse matrix with custom connectivity
matrix = DenSparseMatrix(
    input_size=10,
    output_size=5,
    mapping_width=6,
    mapping_function=mapping_func
)
```

## Testing

Run tests with:
```bash
pytest
```
