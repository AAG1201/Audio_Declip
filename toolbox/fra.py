import numpy as np

def frana(insig, redundancy):
    return np.fft.fft(np.concatenate((insig, np.zeros(len(insig) * (redundancy - 1)))))



def frsyn(insig, redundancy):
    # Perform inverse DFT
    time_signal = np.fft.ifft(insig)


    # Postpad the signal to the desired length
    desired_length = len(insig) // redundancy
    
    if desired_length > len(time_signal):
        postpadded_signal = np.pad(time_signal, (0, desired_length - len(time_signal)), mode='constant')
    else:
        postpadded_signal = time_signal

    return postpadded_signal

