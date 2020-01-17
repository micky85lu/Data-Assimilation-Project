import numpy as np
from model import lorenz63_fdm


spn_time = 100
dt = 0.01
spn_ts = np.arange(0, spn_time, dt)

# spin up and get the initial condition for nature run
x0 = np.array([8, 0, 30])
X_forecast = lorenz63_fdm(x0, spn_ts)
X_nature_ini = X_forecast[:,[-1]]

# create nature run
time = 16
ts = np.arange(0, time, dt)
X_nature = lorenz63_fdm(X_nature_ini, ts)
X_nature.shape

# create initial condition for experiment
x0_perturb = x0 + np.random.randn(3)
X_forecast = lorenz63_fdm(x0_perturb, spn_ts)
X_ini = X_forecast[:,[-1]]
X_ini += 10


np.save('./data/X_nature', X_nature)
np.save('./data/X_ini', X_ini)
np.save('./data/time_span', ts)