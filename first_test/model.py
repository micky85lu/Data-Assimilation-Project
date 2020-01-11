import numpy as np
from scipy.integrate import solve_ivp


def lorenz63_func(t, Y, sigma=10, r=28, b=8/3):
    x = Y[0]
    y = Y[1]
    z = Y[2]
    dxdt = sigma * (y - x)
    dydt = r*x - y - x*z
    dzdt = x*y - b*z
    return [dxdt, dydt, dzdt]


def lorenz63(y0, ts, sigma=10, r=28, b=8/3):
    func = lambda t, Y: lorenz63_func(t, Y, sigma, r, b)
    t_start = ts[0]
    t_end = ts[-1]
    return solve_ivp(func, (t_start,t_end), y0, 'RK45', t_eval=ts)


def lorenz63_fdm(x0, size, dt, sigma=10, r=28, b=8/3):
    """Solve lorenz63 with finit difference method"""
    x = np.zeros((size,))
    y = np.zeros((size,))
    z = np.zeros((size,))
    x[0] = x0[0]
    y[0] = x0[1]
    z[0] = x0[2]
    
    for i in range(1, size):
        x[i] = x[i-1] + sigma * (y[i-1] - x[i-1]) * dt
        y[i] = y[i-1] + (r*x[i-1] - y[i-1] - x[i-1]*z[i-1]) * dt
        z[i] = z[i-1] + (x[i-1]*y[i-1] - b*z[i-1]) * dt
    
    return np.vstack((x, y, z)).T