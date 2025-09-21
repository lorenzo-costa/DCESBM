#####################
# misc functions 
#################
import numba as nb
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
