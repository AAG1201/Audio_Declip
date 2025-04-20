import numpy as np

def hard_thresholding(a, k):

    """
    Performs hard thresholding on the Discrete Fourier Transform (DFT) coefficients.
    
    This function keeps only the k largest magnitude coefficients in the frequency 
    domain while maintaining the conjugate symmetry required for a real-valued signal.

    Parameters:
        a (numpy array): Input array of DFT coefficients.
        k (int): Number of largest magnitude coefficients to retain.

    Returns:
        numpy array: Thresholded DFT coefficients with conjugate symmetry preserved.
    """
    

    odd = len(a) % 2

    # Taking only half of the spectrum + DC coefficient
    a = a[:len(a)//2 + 1]
    a[0] = a[0] / 2

    # Sorting the coefficients
    ind = np.argsort(np.abs(a))[::-1]
    s = np.zeros(len(a), dtype=complex)


    if k < len(a):
        s[ind[:k]] = a[ind[:k]]
    else:
        s = a
        print('Warning: Variable k is greater than the length of the DFT coefficients. The coefficients will not be thresholded.')

    # Compute conjugates of selected coefficients
    s[0] = s[0] * 2

    if odd:
        s_conj = np.conj(np.flip(s[1:]))
    else:
        s_conj = np.conj(np.flip(s[1:-1]))

    s = np.concatenate([s, s_conj])

    return s
