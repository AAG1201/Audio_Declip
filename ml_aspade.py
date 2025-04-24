import numpy as np
from typing import Tuple
from toolbox.fra import frana,frsyn
from toolbox.proj_time import proj_time
from toolbox.hard_thresholding import hard_thresholding
from pipeline import predict_with_model, prepare_mask_for_inference

# Evaluation version 
def ml_aspade(data_clipped: np.ndarray,
                   masks: np.ndarray,
                   Ls: int,
                   max_it: int,
                   epsilon: float,
                   r: int,
                   s: int,
                   redundancy: float,
                   loaded_model,
                   device,
                   train_gen_mode: int,
                   eval_mode: int,
                   factor: float) -> Tuple[np.ndarray, dict]:

    max_it = int(max_it)
    x_hat = np.copy(data_clipped)
    zEst_init = frana(x_hat, redundancy)
    zEst = zEst_init

    u = np.zeros(len(zEst), dtype=complex)
    k = s
    cnt = 1
    bestObj = float('inf')

    if eval_mode:

        zEst_init_real = np.real(zEst_init)
        zEst_init_imag = np.imag(zEst_init)
        zEst_init = np.concatenate([zEst_init_real.flatten(), zEst_init_imag.flatten()], axis=0).reshape(1, -1)

        processed_mask = prepare_mask_for_inference(masks).to(device)

        # Run the model
        zEst, k = predict_with_model(loaded_model, zEst_init, processed_mask)

        zEst = zEst.flatten()

    
        real = zEst[:500]
        imag = zEst[500:]
        zEst = real + 1j * imag  # 👈 no .numpy() needed

        k = int(factor*k)


    # Initialize tracking
    obj_history = []
    sparsity_history = []
    # Dynamic sparsity parameters
    obj_his = np.zeros((3,1))   # Store last 3 objective values
    imp_thres = 1e-4    # Minimum improvement threshold
    max_sparsity = int(len(zEst) * 0.5)  # Maximum sparsity limit (50% of coefficients)




    while cnt <= max_it:
        # set all but k largest coefficients to zero (complex conjugate pairs are taken into consideration)


        z_bar = hard_thresholding(zEst + u, k)

        objVal = np.linalg.norm(zEst - z_bar)  # update termination function

        obj_history.append(objVal)
        sparsity_history.append(k)
        

        # Store objective value history
        obj_his = np.roll(obj_his, 1)
        obj_his[0] = objVal
        
        if objVal <= bestObj:
            bestObj = objVal
            best_x_hat = np.copy(x_hat)
            best_k = k

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

        if cnt > 1:
            adap_epsilon = epsilon * (1 + 0.1 * np.log(cnt))
        else:
            adap_epsilon = epsilon    # termination step with adaptive threshold

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

    if train_gen_mode:
        mask_features = []
        for mask_key in ['Mh', 'Ml', 'Mr']:
            if mask_key in masks:
                mask_features.append(masks[mask_key].astype(float))

        input_features = {
                'frequency_domain': zEst_init,  # Complex input
                'masks': np.concatenate(mask_features),  # Mask features 
            }
        
        
        metrics = {
            'iterations': cnt + 1,
            'final_objective': objVal,
            'best_objective': bestObj,
            'objective_history': obj_history,
            'sparsity_history': sparsity_history,
            'final_sparsity': k,
            'best_sparsity': best_k,
            'best_estimate': zEst + u,
            'initial_estimate': input_features
        }

        return best_x_hat if best_x_hat is not None else x_hat, metrics, cnt

    return x_hat, cnt

