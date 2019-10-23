import numpy as np


def nmc(t1, t2, ndim, obs_error, alpha, model, da, maxiter=5):
    """
    Using NMC method to construct background error covariance matrix.
    Parrish and Derber 1992, MWR
    """
    Pb = np.zeros((ndim,ndim))
    np.fill_diagonal(Pb, obs_error)
    
    Pb_history = np.zeros((maxiter, Pb.size))
    
    for idx in range(maxiter):
        pass