import numpy as np


def OI(xb, yo, Pb, R, H_func=None, H=None):
    """
    Optimal interpolation.
    
    Parameters:
    ----------
    xb: 1d array, shape=(n,)
        State vector of background with n dimension.
    yo: 1d array, shape=(p,)
        Observation. 'p' is number of observations.
    Pb: 2d array, shape=(n,n)
        Background error covariance.
    R: 2d array, shape=(p,p)
        Observation error covariance.
    H_func: optional. callable function
        Observational operator.
        Its input and output are shape (n,) and (p,) array.
        It is necessary if n != p.
    H: optional. 2d array, shape=(n,p)
        Jocobian of H_func.
        It is necessary if n != p.
        
    Return:
    ------
    xa: 1d array, shape=(n,)
        State vector of analysis.
    Pa: 2d array, shape=(n,n)
        Analysis error covariance.
    """
    if H_func is None:
        if len(xb) != len(yo):
            raise ValueError(f"len(xb) should equal to len(yo): {len(xb)} != {len(yo)}")
        K = Pb @ np.linalg.inv(Pb + R)
        xa = xb + K @ (yo - xb)
        Pa = (np.eye(len(xb)) - K) @ Pb
    else:
        K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
        xa = xb + K @ (yo - H_func(xb))
        Pa = (np.eye(len(xb)) - K @ H) @ Pb
    
    return (xa, Pa)