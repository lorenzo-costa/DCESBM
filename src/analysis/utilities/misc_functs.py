#####################
# misc functions 
#################
import numba as nb
import scipy as sc
import numpy as np


def compute_precision(true, preds):
    try:
        return len(set(true).intersection(set(preds)))/len(preds)
    except ZeroDivisionError:
        return 0.0

def compute_recall(true, preds):
    try:
        return len(set(true).intersection(set(preds)))/len(true)
    except ZeroDivisionError:
        return 0.0

# some fast functions
@nb.jit(nopython=True)
def gammaln_nb(x:int)->float:
    return sc.gammaln(x)

@nb.jit(nopython=True)
def rising_factorial(a, n):
    if n == 0:
        return 1
    return a*rising_factorial(a+1, n-1)

@nb.jit(nopython=True)
def truncated_factorial(n, k):
    result = 1
    for i in range(k):
        result *= (n-i)
    return result

def relabel_clusters(cluster_indices):
    unique_labels, counts = np.unique(cluster_indices, return_counts=True) 
    sorted_labels = unique_labels[np.argsort(-counts)] 
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}  # Create mapping
    relabeled_clusters = np.vectorize(label_mapping.get)(cluster_indices)  # Apply mapping

    return relabeled_clusters


@nb.jit(nopython=True, parallel=False)
def compute_co_clustering_matrix(mcmc_draws_users):
    """
    Compute co-clustering matrix from MCMC draws.
    """
    n_iters, num_users = mcmc_draws_users.shape
    
    co_clustering_matrix_users = np.zeros((num_users, num_users))
    for it in nb.prange(n_iters):
        for user_one in range(num_users):
            for user_two in range(num_users):
                if mcmc_draws_users[it, user_one] == mcmc_draws_users[it, user_two]:
                    co_clustering_matrix_users[user_one, user_two] += 1

    return co_clustering_matrix_users
