import numpy as np
from scipy import optimize
from toolbox.hard_clip import hard_clip
from toolbox.sdr import sdr

def clip_sdr(signal, desired_sdr):
    """
    Clips the input signal according to the desired SDR.
    
    Equivalent to the original MATLAB function:
    function [clipped, masks, clipping_threshold, TrueSDR, percentage] = clip_sdr(signal, desiredSDR)
    
    Parameters:
        signal (numpy.ndarray): Input signal.
        desired_sdr (float): Desired Signal-to-Distortion Ratio.
    
    Returns:
        clipped (numpy.ndarray): Clipped signal.
        masks (dict): Masks for clipped and unclipped regions.
        clipping_threshold (float): Optimal clipping threshold.
        true_sdr (float): Actual SDR value after clipping.
        percentage (float): Percentage of clipped samples.
    """
    # Define difference function (equivalent to MATLAB's diffSDR)
    def diff_sdr(threshold):
        clipped_signal, _ = hard_clip(signal, -threshold, threshold)
        return sdr(signal, clipped_signal) - desired_sdr
    
    # Find the optimal clipping threshold for given desired_sdr
    # Using scipy.optimize.bisect or root_scalar for more robust root finding
    eps = np.finfo(float).eps  # Machine epsilon (very small number)
    max_threshold = 0.99 * np.max(np.abs(signal))
    
    # Initial values check to ensure solutions exist in the interval
    sdr_at_min = diff_sdr(eps)
    sdr_at_max = diff_sdr(max_threshold)
    
    # If values have opposite signs, we can find a root
    if sdr_at_min * sdr_at_max <= 0:
        try:
            # Use root_scalar with bisect method for better reliability
            result = optimize.root_scalar(
                diff_sdr,
                bracket=[eps, max_threshold],
                method='bisect'
            )
            clipping_threshold = result.root
        except Exception as e:
            print(f"Optimization error: {e}")
            # Fallback: use linear interpolation to estimate threshold
            if abs(sdr_at_min) < abs(sdr_at_max):
                clipping_threshold = eps
            else:
                clipping_threshold = max_threshold
    else:
        # If no solution in range, pick the threshold that gives closest SDR
        if abs(sdr_at_min) < abs(sdr_at_max):
            clipping_threshold = eps
        else:
            clipping_threshold = max_threshold
        print(f"Warning: Could not find threshold for desired SDR={desired_sdr}. Using closest available.")
    
    # Apply clipping with the found threshold
    clipped, masks = hard_clip(signal, -clipping_threshold, clipping_threshold)
    
    # Calculate the true SDR directly using the clipped signal
    true_sdr = sdr(signal, clipped)
    
    # Computing the percentage of clipped samples
    percentage = (np.sum(masks['Mh']) + np.sum(masks['Ml'])) / len(signal) * 100
    

    
    return clipped, masks, clipping_threshold, true_sdr, percentage