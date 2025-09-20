
from numba import cuda
import numpy as np


def compute_prob_gpu(probs, 
                          mhk_minus, 
                          frequencies_primary_minus, 
                          frequencies_secondary,
                          y_values, 
                          epsilon, 
                          a, 
                          b, 
                          max_clusters, 
                          degree_corrected,
                          degree_cluster_minus, 
                          degree_node, 
                          degree_param, 
                          is_user_mode):
    
    """Function organizing GPU computation."""

    d_log_probs = cuda.to_device(np.zeros_like(probs))
    d_mhk_minus = cuda.to_device(mhk_minus)
    d_frequencies_primary_minus = cuda.to_device(frequencies_primary_minus)
    d_frequencies_secondary = cuda.to_device(frequencies_secondary)
    d_y_values = cuda.to_device(y_values)
    d_degree_cluster_minus = cuda.to_device(degree_cluster_minus)

    # not sure how to specify this part, for now leave 128
    threads_per_block = 128
    blocks_per_grid = (max_clusters + threads_per_block - 1) // threads_per_block

    compute_prob_kernel[blocks_per_grid, threads_per_block](d_log_probs, d_mhk_minus, d_frequencies_primary_minus, 
                                                            d_frequencies_secondary,d_y_values, epsilon, a, b, max_clusters, 
                                                            degree_corrected, d_degree_cluster_minus, degree_node, degree_param, 
                                                            is_user_mode)
    
    if len(probs) > max_clusters:
        compute_extra_prob_kernel[1, 1](d_log_probs, d_frequencies_secondary, d_y_values, epsilon, a, b, max_clusters, 
                                        degree_corrected, degree_node, degree_param)
        
    # Copy log_probs back to host for final calculations
    log_probs = d_log_probs.copy_to_host()

    return log_probs


@cuda.jit
def compute_prob_kernel(log_probs, 
                        mhk_minus, 
                        frequencies_primary_minus, 
                        frequencies_secondary,
                        y_values, 
                        epsilon, 
                        a, 
                        b, 
                        max_clusters, 
                        degree_corrected,
                        degree_cluster_minus, 
                        degree_node, 
                        degree_param, 
                        is_user_mode):
    
    """GPU kernel to compute sampling probabilities."""
    
    i = cuda.grid(1)

    if i < max_clusters:
        p_i = 0.0
        freq_i = frequencies_primary_minus[i]
        a_plus_epsilon = a + epsilon

        for j in range(len(frequencies_secondary)):
            if is_user_mode is True:
                h, k = i, j
            else:
                k, h = i, j
                
            mhk_val = mhk_minus[h, k]
            y_val = y_values[j]

            mhk_plus_a = mhk_val + a_plus_epsilon
            mhk_plus_y_plus_a = mhk_val + y_val + a_plus_epsilon

            log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
            log_freq_prod2 = np.log(b + (freq_i + 1) * frequencies_secondary[j])

            p_i += (cuda.libdevice.lgamma(mhk_plus_y_plus_a) - cuda.libdevice.lgamma(mhk_plus_a) +
                (mhk_plus_a - epsilon) * log_freq_prod1 -(mhk_plus_y_plus_a - epsilon) * log_freq_prod2)

        log_probs[i] += p_i
        
        if degree_corrected is True:
            first = cuda.libdevice.lgamma(frequencies_primary_minus[i]*degree_param + degree_cluster_minus[i])
            second = cuda.libdevice.lgamma((frequencies_primary_minus[i]+1)*degree_param+degree_cluster_minus[i]+degree_node)
            
            third = cuda.libdevice.lgamma((frequencies_primary_minus[i]+1)*degree_param)
            fourth = cuda.libdevice.lgamma(frequencies_primary_minus[i]*degree_param)
            
            fifth = (degree_cluster_minus[i]+degree_node)*np.log(frequencies_primary_minus[i]+1)
            sixth = degree_cluster_minus[i]*np.log(frequencies_primary_minus[i])
            
            log_probs[i] += (first - second + third - fourth + fifth - sixth)
            

@cuda.jit
def compute_extra_prob_kernel(log_probs, 
                              frequencies_secondary, 
                              y_values, 
                              epsilon, 
                              a, 
                              b, 
                              max_clusters, 
                              degree_corrected, 
                              degree_node, 
                              degree_param):
    
    """GPU kernel to compute the probability for a new cluster (H+1 case)."""
    
    p_new = 0.0
    a_plus_epsilon = a + epsilon
    lgamma_a_log_b = - cuda.libdevice.lgamma(a) + a * np.log(b)

    for j in range(len(frequencies_secondary)):
        y_val = y_values[j]
        p_new += (cuda.libdevice.lgamma(y_val + a_plus_epsilon) + lgamma_a_log_b - 
               (y_val + a) * np.log(b + frequencies_secondary[j]))
        log_probs[max_clusters] += p_new
        
        if degree_corrected is True:
            log_probs[max_clusters] += (cuda.libdevice.lgamma(degree_param)-cuda.libdevice.lgamma(degree_param+degree_node))
