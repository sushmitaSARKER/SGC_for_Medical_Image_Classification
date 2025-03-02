import torch

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to a numpy array."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Convert a flattened numpy array `x` to a torch tensor with the given `shape`."""
    return torch.from_numpy(x.reshape(shape))

def get_lr(optimizer):
    """Get the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_data_inverse_scaler(data_centered):
    """
    Get an inverse scaler function to normalize data.

    Args:
        data_centered (bool): If True, rescales data from [-1, 1] to [0, 1]. Otherwise, returns the input as is.

    Returns:
        callable: The inverse scaling function.
    """
    if data_centered:
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x