import numpy as np

def peak_normalize(g, dim=None, L=None):
    """
    Perform peak normalization (infinity norm) of a multi-dimensional array.
    Args:
        g (numpy.ndarray): Input array to normalize.
        dim (int): Dimension to normalize along. If None, it defaults to the first non-singleton dimension.
        L (int): Desired length for the first dimension of the output array. If None, it defaults to the current length.
    Returns:
        numpy.ndarray: Peak-normalized array.
    """
    D = g.ndim  # Number of dimensions
    
    # Determine the dimension to normalize along
    if dim is None:
        dim = 0
        if np.sum(np.array(g.shape) > 1) == 1:  # Check if g is a vector
            dim = np.argmax(np.array(g.shape) > 1)
    
    # Permute the array to bring the desired dimension to the front
    if dim > 0:
        order = [dim] + [d for d in range(D) if d != dim]
        g = np.transpose(g, order)
    
    Ls = g.shape[0]
    
    # Set L to the length of the transform if it's not provided
    if L is None:
        L = Ls
    
    # Remember the exact size for later and modify it for the new length
    permuted_size = list(g.shape)
    permuted_size[0] = L
    
    # Reshape g to a 2D matrix
    if g.size > 0:
        g = g.reshape(Ls, -1)
    W = g.shape[1]
    
    # Normalize each column using the infinity norm
    fnorm = np.zeros(W)
    for ii in range(W):
        fnorm[ii] = np.linalg.norm(g[:, ii], np.inf)
        if fnorm[ii] > 0:
            g[:, ii] = g[:, ii] / fnorm[ii]
    
    # Reshape back to the original dimensions
    g = g.reshape(permuted_size)
    
    # Undo the permutation
    if dim > 0:
        reverse_order = np.argsort(order)
        g = np.transpose(g, reverse_order)
    
    return g