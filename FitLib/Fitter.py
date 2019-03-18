# FitLib/Fitter.py


# ----------------
# Module Docstring
# ----------------

""" Main Fitter class. """


# -------
# Imports
# -------

import math

import numpy as np

from scipy.optimize import minimize

from FitLib.Function import Function


# ------------
# Fitter Class
# ------------

class Fitter:
    """ Main class for fitting datasets. """
    
    # -----------
    # Constructor
    # -----------
    
    def __init__(self, data_or_data_sets, funcs_or_func_sets, fit_range = None):
        """
        Constructor.
        
        Args:
            data_or_data_sets -- (x, y) tuple of list of (x, y) tuples containing data sets to be fitted
            func_or_func_sets -- (list of) Function object(s) or iterable(s) of Function objects to be fitted against each data set
            fit_range -- (x_min, x_max) range over which to fit data sets (default: None)
        """
        
        # Check and convert data to NumPy arrays.
        
        assert data_or_data_sets is not None
        
        shape = np.shape(data_or_data_sets)
        
        if len(shape) == 2:
            # Assume data_or_data_sets is a single data set.
            
            data_or_data_sets = [data_or_data_sets]
        
        data_sets = []
            
        for x, y in data_or_data_sets:
            x = np.array(x, dtype = np.float64)
            y = np.array(y, dtype = np.float64)
            
            n_x, = x.shape
            n_y, = y.shape
        
            assert n_x > 0 and n_x == n_y
            
            data_sets.append((x, y))
        
        self._data_sets = data_sets
        
        # Check and store fitting functions.
        
        assert funcs_or_func_sets is not None
        
        if len(data_sets) == 1:
            # Wrap funcs_or_func_sets in a list.
            
            funcs_or_func_sets = [funcs_or_func_sets]
        
        func_sets = []
        
        for func_or_func_set in funcs_or_func_sets:
            if isinstance(func_or_func_set, Function):
                # func_or_func_set is a single Function object -> wrap in a list.
                
                func_sets.append([func_or_func_set])
            else:
                # Assume func_or_func_set is an iterable over Function objects.
                
                func_set = [
                    func for func in func_or_func_set
                    ]
                
                for func in func_set:
                    assert isinstance(func, Function)
                
                func_sets.append(func_set)
        
        assert len(func_sets) == len(data_sets)

        self._func_sets = func_sets
        
        # Check and store fit range, if supplied.
        
        self._fit_range = None
        
        if fit_range is not None:
            self.FitRange = fit_range
        
        # Field to hold fit result.
        
        self._fit_res = None

    # ---------------
    # Private Methods
    # ---------------

    def _GetParamList(self):
        """ Build a list of unique Parameter objects to be fitted across all sets of functions. """
        
        params = []
        
        for func_set in self._func_sets:
            for func in func_set:
                for param in func.GetFitParams():
                    # Collect a list of unique Parameter objects to be fitted.
                    # This allows for some Functions to transparently share Parameter objects.
                    
                    if param not in params:
                        params.append(param)
        
        return params
    
    def _FitFunction(self, param_vals):
        """ Update parameters across all sets of functions to be fitted and return root-mean-square error averaged over datasets. """
        
        # Update fit function parameters and return the root-mean-square (RMS) error averaged over data sets.
        
        params = self._GetParamList()
        
        assert len(params) == len(param_vals)
        
        for param, val in zip(params, param_vals):
            param.Value = val
        
        return np.mean(
            self.EvaluateFitError()
            )

    # ----------
    # Properties
    # ----------

    @property
    def FitRange(self):
        """ Get the range of x values used for fitting: an (x_min, x_max) tuple of None. """
        
        return self._fit_range
    
    @FitRange.setter
    def FitRange(self, fit_range):
        """ Set the range of x values used for fitting: either a set of (x_min, x_max) values or None. """
        
        if fit_range is not None:
            # Convert to a tuple of (x_min, x_max) values.
            
            x_min, x_max = fit_range
            
            fit_range = (
                float(x_min), float(x_max)
                )
        
        self._fit_range = fit_range
    
    @property
    def FitResult(self):
        """ Get the fit result returned by scipy.optimize.minimize(). """
        
        return self._fit_res
    
    # --------------
    # Public Methods
    # --------------
    
    def Fit(self, method = None, tol = None):
        """
        Fit data.
        
        Args:
            method -- solver used by scipy.optimize.minimize() (default: None)
            tol -- tolerance for fitting (default: None)
        """
        
        params = self._GetParamList()
        
        fit_p_init, fit_p_bounds = [], []
        
        for param in params:
            fit_p_init.append(param.Value)
            fit_p_bounds.append(param.Bounds)
        
        fit_res = minimize(self._FitFunction, fit_p_init, method = method, bounds = fit_p_bounds, tol = tol)
        
        self._fit_res = fit_res
        
        return fit_res.fun
    
    def EvaluateFit(self):
        """ Evaluate current fit(s) and return a list of (x, y, y_fit) tuples for each data set. """
        
        fit_eval = []
        
        for (x, y), func_set in zip(self._data_sets, self._func_sets):
            # If _fit_range is set, adjust the data if required.
            
            if self._fit_range is not None:
                x_min, x_max = self._fit_range
                
                mask = np.logical_and(
                    x >= x_min, x <= x_max
                    )
                
                assert mask.sum() > 0
                
                x, y = x[mask], y[mask]
            
            y_fit = np.zeros_like(x)
        
            for func in func_set:
                y_fit += func.Evaluate(x)
            
            fit_eval.append((x, y, y_fit))
        
        return fit_eval
    
    def EvaluateFitError(self):
        """ Calculate the root-mean-square (RMS) error for the current fit(s). """
        
        rms_errors = []
        
        for _, y, y_fit in self.EvaluateFit():
            rms_errors.append(
                math.sqrt(np.mean((y - y_fit) ** 2))
                )
        
        return rms_errors
    
    def EvaluateNew(self, x):
        """
        Evaluate current fit(s) on a new set of x values.
        
        Notes:
            This method ignores the fitting range, if specified.
        """
        
        # Check and convert x to a NumPy array.
        
        x = np.array(x, dtype = np.float64)
        
        n_x, = x.shape
        
        assert n_x > 0
        
        # Evaluate current fits on new x values.
        
        fit_eval = []
        
        for func_set in self._func_sets:
            y_fit = np.zeros_like(x)
            
            for func in func_set:
                y_fit += func.Evaluate(x)
            
            fit_eval.append(y_fit)
        
        return fit_eval
    
    def GetFitParams(self):
        """ Collect parameter dictionaries for functions used to fit each data set. """
        
        return [
            [func.GetParamsDict() for func in func_set]
                for func_set in self._func_sets
            ]
        
