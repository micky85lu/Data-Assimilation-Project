import pickle
import numpy as np
from scipy.stats import kurtosis
from scipy.special import erf, erfinv


def invcdf(x, mean, var, epsilon, delta):
    """inverse CDF of sinh-arcsinh transform of normal distrubution"""
    return np.sinh(epsilon/delta + 1/delta * np.arcsinh(mean + np.sqrt(2*var) * erfinv(2*x-1)))

def gen_kurtosis_normal(size, mean, var, epsilon, delta):
    u = np.random.rand(size)
    samples = invcdf(u, mean, var, epsilon, delta)
    return samples

def est_tvar(delta, size=1000, times=1000):
    """estimate theory variance"""
    variances = np.zeros((times,))
    for i in range(times):
        u = np.random.rand(size)
        samples = invcdf(u, 0, 2, 0, delta)
        variances[i] = samples.var()
    return np.mean(variances)


# load data and observation settings
X_nature = np.load('./data/X_nature.npy')
ts = np.load('./data/time_span.npy')

dt = 0.01
time = 16
obs_timeintv = 0.08
obs_intv = int(obs_timeintv / dt)
cycle_num = int(time / obs_timeintv)


### generate kurtosis observations

ex_delta = [
    0.5, 0.6, 0.7, 0.8,    # reject H0 
    0.9, 1.2,              # accept H0
    1.6, 1.8, 2, 2.2       # reject H0
]
filename = 'obs_kurtosis_050_220'
#ex_delta = [0.1, 0.2, 0.3, 0.4, 2.5, 2.8, 3.1, 3.4]
#filename = 'obs_kurtosis_010_340'

ex_obs_dict = {}

for ex_d in ex_delta:
    obs_mean = [0, 0, 0]
    obs_var = [2, 2, 2]
    
    obs_timeintv = 0.08
    obs_intv = int(obs_timeintv / dt)
    cycle_num = int(time / obs_timeintv)
    
    size = int(time/dt)
    X_obs_err = np.zeros((3, size))
    for irow, (obsm, obsv) in enumerate(zip(obs_mean, obs_var)):
        # generate observations
        kurt_obs = gen_kurtosis_normal(size, obsm, obsv, 0, ex_d)
        # adjust to given obs variance
        kurt_obs = kurt_obs * np.sqrt(obsv / est_tvar(ex_d))
        
        # set observations as 0 if it is not at the time of assimilation
        kurt_obs_c = kurt_obs.copy()
        kurt_obs_c[::obs_intv] = 0
        kurt_obs = kurt_obs - kurt_obs_c
        X_obs_err[irow,:] = kurt_obs

    ex_obs = X_nature + X_obs_err
    ex_obs = ex_obs[:,::obs_intv]
    
    key = f'{ex_d:.1f}'
    ex_obs_dict[key] = ex_obs
    
# save
fullfn = f'./data/{filename}.pickle'
pickle.dump(ex_obs_dict, open(fullfn, 'wb'))