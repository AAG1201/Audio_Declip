import numpy as np

from toolbox.fra import frana,frsyn
from toolbox.hard_thresholding import hard_thresholding
from toolbox.proj_time import proj_time
from time import time

def dynamic_aspade(data_clipped,  masks, Ls, max_it, epsilon, r, s, redundancy):

    k_arr = []
    objVal_arr = []

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

    start_time = time()

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

        k_arr.append(k)
        objVal_arr.append(objVal)

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

        # if objVal <= adap_epsilon:
        #     break

        if objVal <= epsilon:
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

    processing_time = time() - start_time

    return data_rec, cnt, processing_time, k_arr, objVal_arr



# def dynamic_aspade(data_clipped, masks, Ls, max_it, epsilon, r, s, redundancy):
#     """
#     Focus on adaptive termination while keeping sparsity increases conservative
#     """
#     k_arr = []
#     objVal_arr = []
    
#     max_it = int(max_it)
#     x_hat = np.copy(data_clipped)
#     zEst = frana(x_hat, redundancy)
#     u = np.zeros(len(zEst))
#     k = s
#     cnt = 1
#     bestObj = float('inf')
    
#     # Adaptive termination parameters
#     obj_history = []
#     min_improvement_rate = 1e-6
#     stagnation_window = 5
    
#     start_time = time()
    
#     while cnt <= max_it:
#         z_bar = hard_thresholding(zEst + u, k)
#         objVal = np.linalg.norm(zEst - z_bar)
        
#         if objVal <= bestObj:
#             data_rec = x_hat
#             bestObj = objVal
        
#         k_arr.append(k)
#         objVal_arr.append(objVal)
#         obj_history.append(objVal)
        
#         # Adaptive termination
#         if len(obj_history) >= stagnation_window:
#             # Check improvement rate over recent iterations
#             recent_objs = obj_history[-stagnation_window:]
#             improvement_rate = (recent_objs[0] - recent_objs[-1]) / stagnation_window
            
#             # Early termination if improvement is too slow
#             if improvement_rate < min_improvement_rate and objVal < 10 * epsilon:
#                 break
        
#         # Standard termination
#         if objVal <= epsilon:
#             break
        
#         # Standard ASPADE steps
#         b = z_bar - u
#         syn = frsyn(b, redundancy)
#         syn = syn[:Ls]
#         x_hat = proj_time(syn, masks, data_clipped)
        
#         zEst = frana(x_hat, redundancy)
#         u = u + zEst - z_bar
#         cnt += 1
        
#         # Conservative sparsity increase (closer to original)
#         if cnt % r == 0:
#             k += s  # Keep original increment
    
#     processing_time = time() - start_time
#     return data_rec, cnt, processing_time, k_arr, objVal_arr




# def dynamic_aspade(data_clipped, masks, Ls, max_it, epsilon, r, s, redundancy, 
#                     early_stop_thresh=0.01, adaptive_k=True, cache_transforms=True):
#     """
#     Optimized ASPADE algorithm with multiple speedup strategies
    
#     Args:
#         early_stop_thresh: Early stopping threshold for objVal improvement
#         adaptive_k: Use adaptive k increment based on convergence rate
#         cache_transforms: Cache frame analysis results when possible
#     """
#     k_arr = []
#     objVal_arr = []
    
#     # initialization of variables
#     max_it = int(max_it)
#     x_hat = np.copy(data_clipped)
#     zEst = frana(x_hat, redundancy)
#     u = np.zeros(len(zEst))
#     k = s
#     cnt = 1
#     bestObj = float('inf')
#     sdr_iter = np.full((max_it, 1), np.nan)
#     obj_iter = np.full((max_it, 1), np.nan)
    
#     # Optimization variables
#     prev_objVal = float('inf')
#     stagnation_count = 0
#     k_increment_factor = 1.0
    
#     # Cache for frame operations (if transforms are expensive)
#     frame_cache = {} if cache_transforms else None
    
#     start_time = time()
    
#     while cnt <= max_it:
        
#         # Strategy 1: Use cached transforms if available
#         cache_key = hash(x_hat.tobytes()) if cache_transforms else None
#         if cache_transforms and cache_key in frame_cache:
#             zEst = frame_cache[cache_key]
#         else:
#             zEst = frana(x_hat, redundancy)
#             if cache_transforms:
#                 frame_cache[cache_key] = zEst.copy()
        
#         z_bar = hard_thresholding(zEst + u, k)
#         objVal = np.linalg.norm(zEst - z_bar)
        
#         if objVal <= bestObj:
#             data_rec = x_hat
#             bestObj = objVal
            
#         k_arr.append(k)
#         objVal_arr.append(objVal)
        
#         # Strategy 2: Early stopping based on convergence rate
#         improvement_rate = (prev_objVal - objVal) / prev_objVal if prev_objVal > 0 else 1.0
        
#         if objVal <= epsilon:
#             break
            
#         # Strategy 3: Adaptive early stopping if improvement is minimal
#         if improvement_rate < early_stop_thresh:
#             stagnation_count += 1
#             if stagnation_count >= 3:  # Allow 3 consecutive poor improvements
#                 break
#         else:
#             stagnation_count = 0
        
#         # projection onto the set of feasible solutions 
#         b = z_bar - u
#         syn = frsyn(b, redundancy)
#         syn = syn[:Ls]
#         x_hat = proj_time(syn, masks, data_clipped)
        
#         # dual variable update
#         zEst = frana(x_hat, redundancy)
#         u = u + zEst - z_bar
        
#         cnt += 1
#         prev_objVal = objVal
        
#         # Strategy 4: Adaptive k increment based on convergence behavior
#         if cnt % r == 0:
#             if adaptive_k:
#                 # Increase k more aggressively if convergence is slow
#                 if improvement_rate < 0.05:
#                     k_increment_factor = min(2.0, k_increment_factor * 1.2)
#                 else:
#                     k_increment_factor = max(1.0, k_increment_factor * 0.9)
#                 k += int(s * k_increment_factor)
#             else:
#                 k += s
    
#     processing_time = time() - start_time
#     return data_rec, cnt, processing_time, k_arr, objVal_arr







# def dynamic_aspade(data_clipped, masks, Ls, max_it, epsilon, r, s, redundancy):
#     """
#     Optimized warm start that reduces cycles while maintaining speed
#     Key: Better termination of warm start phase  no redcution in cyle but SDR improve
#     """
#     k_arr = []
#     objVal_arr = []
    
#     max_it = int(max_it)
    
#     # Optimized warm start with early termination
#     warm_iterations = min(3, max_it // 4)  # Limit warm start iterations
    
#     if warm_iterations > 0:
#         warm_redundancy = max(1, redundancy // 2)
#         x_warm = np.copy(data_clipped)
#         zEst_warm = frana(x_warm, warm_redundancy)
#         u_warm = np.zeros(len(zEst_warm))
#         k_warm = s
#         prev_warm_obj = float('inf')
        
#         for i in range(warm_iterations):
#             z_bar_warm = hard_thresholding(zEst_warm + u_warm, k_warm)
#             warm_objVal = np.linalg.norm(zEst_warm - z_bar_warm)
            
#             # Early termination if warm start converges
#             if warm_objVal < epsilon * 2 or (prev_warm_obj - warm_objVal) / prev_warm_obj < 0.01:
#                 break
                
#             b_warm = z_bar_warm - u_warm
#             syn_warm = frsyn(b_warm, warm_redundancy)
#             syn_warm = syn_warm[:Ls]
#             x_warm = proj_time(syn_warm, masks, data_clipped)
#             zEst_warm = frana(x_warm, warm_redundancy)
#             u_warm = u_warm + zEst_warm - z_bar_warm
#             prev_warm_obj = warm_objVal
        
#         x_hat = x_warm
#     else:
#         x_hat = np.copy(data_clipped)
    
#     # Main algorithm with cycle reduction
#     zEst = frana(x_hat, redundancy)
#     u = np.zeros(len(zEst))
#     k = s
#     cnt = 1
#     bestObj = float('inf')
    
#     start_time = time()
    
#     while cnt <= max_it:
        
#         z_bar = hard_thresholding(zEst + u, k)
#         objVal = np.linalg.norm(zEst - z_bar)
        
#         if objVal <= bestObj:
#             data_rec = x_hat
#             bestObj = objVal
            
#         k_arr.append(k)
#         objVal_arr.append(objVal)
        
#         if objVal <= epsilon:
#             break
        
#         # projection onto the set of feasible solutions 
#         b = z_bar - u
#         syn = frsyn(b, redundancy)
#         syn = syn[:Ls]
#         x_hat = proj_time(syn, masks, data_clipped)
        
#         # dual variable update
#         zEst = frana(x_hat, redundancy)
#         u = u + zEst - z_bar
        
#         cnt += 1
        
#         # Original k increment schedule
#         if cnt % r == 0:
#             k += s
    
#     processing_time = time() - start_time
#     return data_rec, cnt, processing_time, k_arr, objVal_arr

