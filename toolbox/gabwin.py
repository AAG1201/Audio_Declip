import numpy as np
from math import gcd, lcm

def gabwin(M, dim=None, L=None):
    """
    Computes a window similar to the MATLAB code provided.
    Args:
        M (int): The number of samples.
        dim (int): The dimension for computation.
        L (int): Desired length of the transform.
    Returns:
        numpy.ndarray: The computed window.
    """
    # Define x based on M
    if M % 2 == 0:  # For even M
        x = np.concatenate((np.arange(0, 0.5, 1/M), np.arange(-0.5, 0, 1/M)))
    else:  # For odd M
        x = np.concatenate((np.arange(0, 0.5 - 0.5/M, 1/M), 
                            np.arange(-0.5 + 0.5/M, 0, 1/M)))
    
    # Compute the window
    g = 0.5 + 0.5 * np.cos(2 * np.pi * x)
    
    # Force the window to 0 outside (-0.5, 0.5)
    g = g * (np.abs(x) < 0.5)
    
    # Default values for L and dim
    if L is None:
        L = len(g)
    
    if dim is None:
        dim = 0
        if np.sum(np.array(g.shape) > 1) == 1:
            dim = np.argmax(np.array(g.shape) > 1)
    
    # Permute dimensions if necessary
    if dim > 0:
        g = np.moveaxis(g, dim, 0)
    
    Ls = g.shape[0]
    permuted_shape = list(g.shape)
    permuted_shape[0] = L
    
    # Reshape g to a 2D matrix
    if g.size > 0:
        g = g.reshape(Ls, -1)
    W = g.shape[1]
    
    # Normalize each column
    fnorm = np.zeros(W)
    for ii in range(W):
        fnorm[ii] = np.linalg.norm(g[:, ii], 2)
        if fnorm[ii] > 0:
            g[:, ii] = g[:, ii] / fnorm[ii]
    
    # Reshape back to the original dimensions
    g = g.reshape(permuted_shape)
    
    # Undo the permutation
    if dim > 0:
        g = np.moveaxis(g, 0, dim)
    
    # Ensure length is a multiple of M
    g = np.concatenate((g, np.zeros(-len(g) % int(M))))
    
    return g

