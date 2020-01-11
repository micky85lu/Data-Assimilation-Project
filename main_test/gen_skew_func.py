import numpy as np
from scipy.stats import skewnorm


def tskew(alpha):
    """calculate theoretical skewness of given alpha"""
    d = alpha / np.sqrt(1+alpha**2)
    return (4-np.pi)/2 * (d*np.sqrt(2/np.pi)) ** 3 / (1-2*d**2/np.pi) ** (3/2)

def gen_skewnormal(mean, var, alpha, size, random_state=None):
    """generate random number by skewnorm, and adjust them into given mean and variance"""
    # generate standard skew normal distribution
    X = skewnorm.rvs(alpha, loc=0, scale=1, size=size, random_state=random_state)
    
    # theory expectation value (mean) and variance of standard skew normal distribution
    tmean = np.sqrt(2/np.pi) * alpha / np.sqrt(1+alpha**2)
    tvar = 1 - 2/np.pi * alpha**2 / (1+alpha**2)

    # adjust var, then adjust mean
    X = np.sqrt(var/tvar) * X
    tmean = np.sqrt(var/tvar) * np.sqrt(2/np.pi) * alpha / np.sqrt(1+alpha**2)
    X = X + mean - tmean
    
    return X

def gen_skew_obs(mean, var, alpha, obs_intv):
    # load data and observation settings
    X_nature = np.load('./data/X_nature.npy')
    ts = np.load('./data/time_span.npy')

    dt = 0.01
    time = 16
    obs_timeintv = 0.08
    obs_intv = int(obs_timeintv / dt)
    
    X_obs_err = np.zeros_like(X_nature)
    for irow in range(X_nature.shape[0]):
        skew_obs = gen_skewnormal(mean, var, alpha, ts.size)
        # set observations as 0 if it is not at the time of assimilation
        skew_obs_c = skew_obs.copy()
        skew_obs_c[::obs_intv] = 0
        skew_obs = skew_obs - skew_obs_c
        X_obs_err[irow,:] = skew_obs
        
    obs = X_nature + X_obs_err
    obs = obs[:,::obs_intv]
    return obs