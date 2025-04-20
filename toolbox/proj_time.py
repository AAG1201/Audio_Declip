import numpy as np


def proj_time(x, masks, data_clipped):
    
    """
    PROJ_TIME projects the input signal vector x onto the set of feasible 
    solutions for the declipping problem in the time domain.

    Parameters:
        x (numpy array): Input signal vector.
        masks (dict): Dictionary containing three logical masks:
            - 'Mr' (numpy array): Mask for reliable (unclipped) samples.
            - 'Mh' (numpy array): Mask for samples clipped at the upper bound.
            - 'Ml' (numpy array): Mask for samples clipped at the lower bound.
        data_clipped (numpy array): The clipped version of the signal.

    Returns:
        numpy array: The projected signal with corrections applied based on masks.
    """



    proj = np.copy(x)
    # Debugging Checks
    assert len(x) == len(data_clipped), "Signal length mismatch"
    assert len(masks['Mr']) == len(x), "Mask 'Mr' length mismatch"
    assert len(masks['Mh']) == len(x), "Mask 'Mh' length mismatch"
    assert len(masks['Ml']) == len(x), "Mask 'Ml' length mismatch"

    proj[masks['Mr']] = data_clipped[masks['Mr']]
    proj[masks['Mh']] = np.maximum(x[masks['Mh']], data_clipped[masks['Mh']])
    proj[masks['Ml']] = np.minimum(x[masks['Ml']], data_clipped[masks['Ml']])

    
    return proj