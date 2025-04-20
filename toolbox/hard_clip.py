import numpy as np

def hard_clip(signal, t_min, t_max):
    """
    Performs hard clipping of input signal and returns clipped signal and masks.

    Parameters:
        signal (numpy.ndarray): Input signal.
        t_min (float): Lower clipping threshold.
        t_max (float): Upper clipping threshold.

    Returns:
        clipped (numpy.ndarray): Clipped signal.
        masks (dict): Dictionary containing three masks:
                      - 'Mh': Mask for values above t_max.
                      - 'Ml': Mask for values below t_min.
                      - 'Mr': Mask for values within the range [t_min, t_max].
    """
    if np.min(signal) >= t_min and np.max(signal) <= t_max:
        print("Warning: Clipping range too large. No clipping will occur!")

    if t_min >= t_max:
        raise ValueError("Lower clipping level must be smaller than the upper one!")

    # Hard clipping & computing masks
    clipped = np.copy(signal)
    masks = {
        'Mh': signal > t_max,
        'Ml': signal < t_min,
        'Mr': ~(signal > t_max) & ~(signal < t_min)
    }

    clipped[masks['Mh']] = t_max
    clipped[masks['Ml']] = t_min

    return clipped, masks