import numpy as np

from toolbox.fra import frana,frsyn
from toolbox.hard_thresholding import hard_thresholding
from toolbox.proj_time import proj_time

def dynamic_aspade(data_clipped,  masks, Ls, max_it, epsilon, r, s, redundancy):
    max_it=int(max_it)
    x_hat = np.copy(data_clipped)
    zEst = frana(x_hat, redundancy)
    u = np.zeros(len(zEst))
    k = s
    cnt = 1
    bestObj = float('inf')

    # Dynamic sparsity parameters
    obj_his = np.zeros((3,1))   # Store last 3 objective values
    imp_thres = 1e-4    # Minimum improvement threshold
    max_sparsity = int(len(zEst) * 0.5)   # Maximum sparsity limit (50% of coefficients)

    while cnt <= max_it:
        # set all but k largest coefficients to zero (complex conjugate pairs are taken into consideration)
        z_bar = hard_thresholding(zEst + u, k)

        objVal = np.linalg.norm(zEst - z_bar)  # update termination function

        # Store objective value history
        obj_his = np.roll(obj_his, 1)
        obj_his[0] = objVal
        
        if objVal <= bestObj:
            data_rec = x_hat
            bestObj = objVal

        # Dynamic sparsity update based on convergence behavior

        if cnt > 3:
            rel_improvement = (obj_his[2] - objVal) / obj_his[2]    # Calculate relative improvement
            
            if rel_improvement < imp_thres:
                k = min(k + 2 * s, max_sparsity)    # Slow convergence - increase sparsity more aggressively
            elif rel_improvement > 5 * imp_thres:
                k = k   # Fast convergence - maintain current sparsity
            else:
                if cnt % r == 0:
                    k = min(k + s, max_sparsity)

        adap_epsilon = epsilon * (1 + 0.1 * np.log(cnt))    # termination step with adaptive threshold

        if objVal <= adap_epsilon:
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
    

    return x_hat, cnt
