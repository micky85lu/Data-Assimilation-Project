import numpy as np


# this Pb is from the result of `lorenz63_nmc_test`
Pb = np.array([
    [ 95.92487549, 108.82721654,  13.52170192],
    [108.82721654, 135.97355097,  18.39642468],
    [ 13.52170192,  18.39642468,  64.98710037]
])
Pb *= 0.02

obs_var = [2, 2, 2]
R = np.zeros((len(obs_var), len(obs_var)))
np.fill_diagonal(R, obs_var)

np.save('./data/R', R)
np.save('./data/Pb', Pb)