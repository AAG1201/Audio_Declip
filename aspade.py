import numpy as np

from toolbox.fra import frana,frsyn
from toolbox.hard_thresholding import hard_thresholding
from toolbox.proj_time import proj_time

def aspade(data_clipped,  masks, Ls, max_it, epsilon, r, s, redundancy):

    # initialization of variables
    max_it=int(max_it)
    x_hat = np.copy(data_clipped)
    zEst = frana(x_hat, redundancy)

    u = np.zeros(len(zEst))
    k = s
    cnt = 1
    bestObj = float('inf')

    sdr_iter = np.full((max_it, 1), np.nan)
    obj_iter = np.full((max_it, 1), np.nan)

    while cnt <= max_it:
        
        z_bar = hard_thresholding(zEst + u, k)  # set all but k largest coefficients to zero (complex conjugate pairs are taken into consideration)

        objVal = np.linalg.norm(zEst - z_bar)  # update termination function
        
        if objVal <= bestObj:
            data_rec = x_hat
            bestObj = objVal
        
        if objVal <= epsilon:   # termination step
            break
        
        # projection onto the set of feasible solutions 
        b = z_bar - u
        syn = frsyn(b, redundancy)
        syn = syn[:Ls]
        x_hat = proj_time(syn, masks, data_clipped)
        
        # dual variable update
        zEst = frana(x_hat, redundancy)
        u = u + zEst - z_bar
        
        cnt += 1    # iteration counter update
        
        # incrementation of variable k (require less sparse signal in next iteration)
        if cnt % r == 0:
            k += s
        

    return x_hat, cnt
