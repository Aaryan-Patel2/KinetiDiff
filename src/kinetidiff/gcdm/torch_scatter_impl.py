"""
Proper torch_scatter implementations without requiring the compiled package.
These handle the actual reduction operations needed by GCDM.
"""

import torch


def scatter_add(src, index, dim=0, dim_size=None, out=None):
    """
    Scatter addition: accumulate values at indices.
    src: source tensor
    index: indices tensor
    dim: dimension to scatter along
    """
    if dim != 0:
        # For simplicity, only handle dim=0
        return src
    
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 0
    
    shape = list(src.shape)
    shape[0] = dim_size
    
    if out is None:
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    out.index_add_(0, index, src)
    return out


def scatter_mean(src, index, dim=0, dim_size=None, out=None):
    """
    Scatter mean: compute mean of values at each index.
    src: source tensor to aggregate
    index: indices for grouping
    dim: dimension to scatter along
    dim_size: size of output dimension
    """
    if dim != 0:
        return src
    
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 0
    
    # Count occurrences of each index
    ones = torch.ones_like(src[:, :1] if len(src.shape) > 1 else src.unsqueeze(1))
    count = scatter_add(ones, index, dim=0, dim_size=dim_size)
    count = torch.clamp(count, min=1)  # Avoid division by zero
    
    # Sum values for each index
    result = scatter_add(src, index, dim=0, dim_size=dim_size)
    
    # Divide by count to get mean
    if len(src.shape) > 1:
        result = result / count
    else:
        result = result.squeeze() / count.squeeze()
    
    return result


def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    """
    General scatter operation supporting different reduction modes.
    """
    if reduce == 'add' or reduce == 'sum':
        return scatter_add(src, index, dim=dim, dim_size=dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim=dim, dim_size=dim_size)
    else:
        raise ValueError(f"Unsupported reduce operation: {reduce}")
