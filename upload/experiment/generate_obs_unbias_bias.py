import pickle
import numpy as np


def gen_random_obserr(mean, var, size, obs_intv, random_state=None):
    """
    Generate random gaussian observation error.
    
    Parameters:
    ----------
    mean, var: scaler.
        Mean and variance for gaussian distribution.
    size: int.
        The size of output array.
    obs_intv: int.
        The observation interval length in the output array.
    random_state: int.
        Random state. Default is None and it will use np.random.randint.
        
    Return:
    ------
    obs_err: 1-d array.
        The array which observation error occurs every `obs_intv` and others are 0.
        
    EX:
    >>> gen_random_obserr(0, 1, 12, 4)
    array([-0.34889445,  0,  0,  0,  0.98370343,  0,  0,  0,
           0.58092283,  0,  0,  0])
    """
    if random_state is None:
        random_state = np.random.randint(0, 50)
    
    length = np.ceil(size/obs_intv) * obs_intv
    obs_err = np.zeros(int(length)).reshape((-1,obs_intv))
    
    rng = np.random.RandomState(random_state)
    obs_err[:,0] = rng.normal(mean, np.sqrt(var), size=obs_err.shape[0])
    obs_err = obs_err.ravel()[:size]
    return obs_err


# load data and observation settings
X_nature = np.load('./data/X_nature.npy')
ts = np.load('./data/time_span.npy')

dt = 0.01
time = 16
obs_timeintv = 0.08
obs_intv = int(obs_timeintv / dt)
cycle_num = int(time / obs_timeintv)


### generate normal observations (unbias)
obs_mean = [0, 0, 0]
obs_var = [2, 2, 2]

X_obs_err = np.zeros((3, ts.size))
for irow, (obsm, obsv) in enumerate(zip(obs_mean, obs_var)):
    X_obs_err[irow,:] = gen_random_obserr(obsm, obsv, ts.size, obs_intv)
    
X_obs = X_nature + X_obs_err
X_obs = X_obs[:,::obs_intv]

np.save('./data/obs_normal', X_obs)


### generate bias observations
experiments = [
    [0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    np.arange(0.2, 5.2+0.25, 0.25)
]
ex_filenames = ['obs_bias_005_040', 'obs_bias_020_520']

for ex_fn, ex_mean in zip(ex_filenames, experiments):
    ex_obs_dict = {}
    
    for ex_m in ex_mean:
        obs_mean = [ex_m for _ in range(3)]
        obs_var = [2, 2, 2]

        X_obs_err = np.zeros((3, ts.size))
        for irow, (obsm, obsv) in enumerate(zip(obs_mean, obs_var)):
            X_obs_err[irow,:] = gen_random_obserr(obsm, obsv, ts.size, obs_intv)

        ex_obs = X_nature + X_obs_err
        ex_obs = ex_obs[:,::obs_intv]

        key = f'{ex_m:4.2f}'
        ex_obs_dict[key] = ex_obs
    
    # save file
    fullfn = f'./data/{ex_fn}.pickle'
    pickle.dump(ex_obs_dict, open(fullfn, 'wb'))