import numpy as np

def sdr(u, v):
    """
    Computes the Signal-to-Distortion Ratio (SDR).

    Parameters:
        u (numpy.ndarray): Original signal.
        v (numpy.ndarray): Processed signal.

    Returns:
        sdr_val (float): SDR value in dB.
    """
    return 20 * np.log10(np.linalg.norm(u) / np.linalg.norm(u - v))