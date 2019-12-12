import warnings
import numpy as np
from scipy.optimize import minimize


class DAbase:
    def __init__(self, model, dt, store_history=False):
        self._isstore = store_history
        self._params = {'alpha': 0, 'inflat': 1}
        self.model = model
        self.dt = dt
        self.X_ini = None
        
    def set_params(self, param_list, **kwargs):
        for key, value in kwargs.items():
            if key in param_list:
                self._params[key] = kwargs.get(key)
            else:
                raise ValueError(f'Invalid parameter: {key}')
        
    def _check_params(self, param_list):
        missing_params = []
        for var in param_list:
            if self._params.get(var) is None:
                missing_params.append(var)
        return missing_params


class ExtendedKF(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
            'H', 
            'M', 
            'alpha', 
            'inflat'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
        if self._params.get('H') is None:
            H = np.eye(self._params.get('R').shape[0])
            self._params['H'] = H
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, Pb, R, H_func=None, H=None):
        if H_func is None:
            K = Pb @ np.linalg.inv(Pb + R)
            xa = xb + K @ (yo - xb)
            Pa = (np.eye(len(xb)) - K) @ Pb
        else:
            K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
            xa = xb + K @ (yo - H_func(xb))
            Pa = (np.eye(len(xb)) - K @ H) @ Pb
        return (xa, Pa)
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini'].copy()
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        H = self._params['H']
        alpha = self._params['alpha']
        inflat = self._params['inflat']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.arange(t_start, cycle_len*dt, dt)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa, Pa = self._analysis(xb, obs[:,[nc]], Pb, R, H_func, H)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            M = self._params['M'](xb[0,0], xb[1,0], xb[2,0])
            Pb = alpha * Pb + (1-alpha) * M @ Pa @ M.T
            Pb *= inflat
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.arange(t_start, t_start+cycle_len*dt, dt)
            
        self.background = background
        self.analysis = analysis
        
        
class OI(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
            'H'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
        if self._params.get('H') is None:
            H = np.eye(self._params.get('R').shape[0])
            self._params['H'] = H
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, Pb, R, H_func=None, H=None):
        if H_func is None:
            K = Pb @ np.linalg.inv(Pb + R)
            xa = xb + K @ (yo - xb)
            Pa = (np.eye(len(xb)) - K) @ Pb
        else:
            K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
            xa = xb + K @ (yo - H_func(xb))
            Pa = (np.eye(len(xb)) - K @ H) @ Pb
        return (xa, Pa)
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini'].copy()
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        H = self._params['H']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.arange(t_start, cycle_len*dt, dt)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa, _ = self._analysis(xb, obs[:,[nc]], Pb, R, H_func, H)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.arange(t_start, t_start+cycle_len*dt, dt)
            
        self.background = background
        self.analysis = analysis
        

class M3DVar(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _3dvar_costfunction(self, x, xb, yo, invPb, invR, H_func=None, H=None):
        """
        x and xb is 1d array with shape (n,), the other is 2d matrix
        """
        x = x[:,np.newaxis]
        xb = xb[:,np.newaxis]

        if H_func is None:
            innovation = yo - x
        else:
            innovation = yo - H_func(x) 

        return 0.5 * (xb-x).T @ invPb @ (xb-x) + 0.5 * innovation.T @ invR @ innovation

    def _analysis(self, xb, yo, Pb, R, H_func=None):    
        if H_func is None:
            innovation = yo - xb
        else:
            innovation = yo - H_func(xb)

        invPb = np.linalg.inv(Pb)
        invR = np.linalg.inv(R)
        cost_func = lambda x: self._3dvar_costfunction(x, xb.ravel(), yo, invPb, invR)

        return minimize(cost_func, xb.ravel(), method='BFGS').x
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini'].copy()
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.arange(t_start, cycle_len*dt, dt)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa = self._analysis(xb, obs[:,[nc]], Pb, R, H_func)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.arange(t_start, t_start+cycle_len*dt, dt)
            
        self.background = background
        self.analysis = analysis
        
        
class EnKF(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ens_ini', 
            'obs', 
            'obs_interv', 
            'R', 
            'H_func', 
            'alpha', 
            'inflat'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, R, H_func=None):
        """xb.shape = (n_dim, n_ens)"""
        if H_func is None:
            H_func = lambda arr: arr
        
        N_ens = xb.shape[1]
        yo_ens = np.random.multivariate_normal(yo.ravel(), R, size=N_ens).T  # (ndim_yo, N_ens)
        xb_mean = xb.mean(axis=1)[:,np.newaxis]  # (ndim_xb, 1)
        
        xa_ens = np.zeros((xb.shape[0], N_ens))
        for iens in range(N_ens):
            xb_mean = xb.mean(axis=1)[:,np.newaxis]
            Xb_perturb = xb - xb_mean
            HXb_perturb = H_func(Xb_perturb) - H_func(Xb_perturb).mean(axis=1)[:,np.newaxis]
            
            PfH_T = Xb_perturb @ HXb_perturb.T / (N_ens-1)
            HPfH_T = HXb_perturb @ HXb_perturb.T / (N_ens-1)
            K = PfH_T @ np.linalg.inv(HPfH_T + R)
            xa_ens[:,[iens]] = xb[:,[iens]] + K @ (yo_ens[:,[iens]] - H_func(xb[:,[iens]]))
            
        return xa_ens
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ens_ini'].copy()
        obs = self._params['obs']
        R = self._params['R']
        H_func = self._params['H_func']
        alpha = self._params['alpha']
        inflat = self._params['inflat']
        
        ndim, N_ens = xb.shape
        background = np.zeros((N_ens, ndim, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.arange(t_start, cycle_len*dt, dt)
        
        for nc in range(cycle_num):
            # analysis
            xa = self._analysis(xb, obs[:,[nc]], R, H_func)
            
            # inflat
            xa_perturb = xa - xa.mean(axis=1)[:,np.newaxis]
            xa_perturb *= inflat
            xa = xa.mean(axis=1)[:,np.newaxis] + xa_perturb
            
            # ensemble forecast
            for iens in range(N_ens):
                x_forecast = model(xa[:,iens], ts)   # (ndim, ts.size)
                
                idx1 = nc*cycle_len
                idx2 = (nc+1)*cycle_len
                analysis[iens,:,idx1:idx2] = x_forecast
                background[iens,:,[idx1]] = xb[:,iens]
                background[iens,:,(idx1+1):idx2] = x_forecast[:,1:]
                
                # xb for next cycle
                xb[:,iens] = x_forecast[:,-1]
                
            # for next cycle
            t_start = int(ts[-1] + dt)
            ts = np.arange(t_start, t_start+cycle_len*dt, dt)
            
        self.background = background
        self.analysis = analysis