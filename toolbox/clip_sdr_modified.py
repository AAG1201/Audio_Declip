import numpy as np

from toolbox.hard_clip import hard_clip
from toolbox.sdr import sdr

def clip_sdr_modified(signal, clipping_threshold):
    """
    Clips the input signal according to the specified threshold 
    and calculates the SDR, percentage of clipped samples, and masks.

    Parameters:
        signal (numpy.ndarray): Input signal.
        clipping_threshold (float): Clipping threshold.

    Returns:
        clipped (numpy.ndarray): Clipped signal.
        masks (dict): Masks for clipped and unclipped regions.
        clipping_threshold (float): Clipping threshold used.
        true_sdr (float): SDR value after clipping.
        percentage (float): Percentage of clipped samples.
    """
    # Clipping the signal with the provided threshold
    clipped, masks = hard_clip(signal, -clipping_threshold, clipping_threshold)

    # Calculating the true SDR after clipping
    true_sdr = sdr(signal, clipped)

    # Computing the percentage of clipped samples
    percentage = (np.sum(masks['Mh']) + np.sum(masks['Ml'])) / len(signal) * 100

    return clipped, masks, clipping_threshold, true_sdr, percentage