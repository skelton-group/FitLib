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
    
    def __init__(self, data_set_or_data_sets, func_or_func_set, fit_range = None):
        """
        Constructor.
        
        Args:
            data_set_or_data_sets -- (x, y) tuple or iterable of (x, y) tuples containing data sets to be fitted
            func_or_func_set -- Function object or iterable of Function objects to be fitted against each data set
            fit_range -- (x_min, x_max) range over which to fit data sets (default: None)
            
            Notes:
                To fit a data set against multiple functions, combine them into a CompositeFunction object first.
        """
        
        # Check and convert data to NumPy arrays.
        
        assert data_set_or_data_sets is not None
        
        shape = np.shape(data_set_or_data_sets)
        
        if len(shape) == 2:
            # Assume data_or_data_sets is a single data set.
            
            data_set_or_data_sets = [data_set_or_data_sets]
        
        data_sets = []
            
        for x, y in data_set_or_data_sets:
            x = np.asarray(x, dtype = np.float64)
            y = np.asarray(y, dtype = np.float64)
            
            n_x, = x.shape
            n_y, = y.shape
        
            assert n_x > 0 and n_x == n_y
            
            data_sets.append((x, y))
        
        self._data_sets = data_sets
        
        # Check and store fitting functions.
        
        func_set = None
        
        if isinstance(func_or_func_set, Function):
            # Single function -> wrap in a list.
            
            func_set = [func_or_func_set]
        else:
            # Assume func_or_func_sets is an iterable of Function objects.
            
            func_set = [func for func in func_or_func_set]
            
            for func in func_set:
                assert isinstance(func, Function)
        
        self._func_set = func_set    
        
        # Check and store fit range, if supplied.
        
        self._fit_range = None
        
        if fit_range is not None:
            self.FitRange = fit_range
        
        # Field to hold fit result.
        
        self._fit_res = None

    # ---------------
    # Private Methods
    # ---------------

    def _GetFitParamsList(self):
        """ Build a flat list of unique Parameter objects to be fitted across all functions. """
        
        params_list = []
        
        for func in self._func_set:
            for param in func.GetParamsList(fit_only = True):
                # Collect a list of unique Parameter objects to be fitted.
                # This allows for some Functions to transparently share Parameter objects.
                
                if param not in params_list:
                    params_list.append(param)
        
        return params_list
    
    def _FitFunction(self, param_vals):
        """ Update parameters across all functions to be fitted and return root-mean-square error averaged over datasets. """
        
        # Update fit function parameters and return the root-mean-square (RMS) error averaged over data sets.
        
        params = self._GetFitParamsList()
        
        assert len(params) == len(param_vals)
        
        for param, val in zip(params, param_vals):
            param.Value = val
        
        return np.mean(
            self.EvaluateFitError(strip_dims = False)
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
    
    def Fit(self, **kwargs):
        """
        Fit data and return root-mean-square (RMS) error averaged over data sets.
        
        Notes:
            Keyword arguments are passed directly to the scipy.optimize.minimize() function.
        """
        
        params = self._GetFitParamsList()
        
        fit_p_init, fit_p_bounds = [], []
        
        for param in params:
            fit_p_init.append(param.Value)
            fit_p_bounds.append(param.Bounds)
        
        fit_res = minimize(self._FitFunction, fit_p_init, bounds = fit_p_bounds, **kwargs)
        
        self._fit_res = fit_res
        
        return fit_res.fun
    
    def EvaluateFit(self, strip_dims = True):
        """
        Evaluate current fit(s) and return a list of (x, y, y_fit) tuples for each data set.
        
        Args:
            strip_dims -- if True (default), return a single (x, y, y_fit) tuple rather than a list if a single data set is being fitted
        """
        
        fit_eval = []
        
        for (x, y), func in zip(self._data_sets, self._func_set):
            # If a fit range has been set, adjust the data if required.
            
            if self._fit_range is not None:
                x_min, x_max = self._fit_range
                
                mask = np.logical_and(
                    x >= x_min, x <= x_max
                    )
                
                assert mask.sum() > 0
                
                x, y = x[mask], y[mask]
            
            y_fit = func.Evaluate(x)
        
            fit_eval.append((x, y, y_fit))
        
        # If strip_dims is set and only a single data set is being fit, return a single (x, y, y_fit) tuple.
        
        if strip_dims and len(fit_eval) == 1:
            return fit_eval[0]
        else:
            return fit_eval
    
    def EvaluateFitError(self, strip_dims = True):
        """
        Calculate the root-mean-square (RMS) error for the current fit(s).
        
        Args:
            strip_dims -- if True (default), return a single RMS value rather than a list if a single data set is being fitted
        """
        
        rms_errors = []
        
        for _, y, y_fit in self.EvaluateFit(strip_dims = False):
            rms_errors.append(
                math.sqrt(np.mean((y - y_fit) ** 2))
                )
        
        # If strip_dims is set and only a single data set is being fit, return a single error value.
        
        if strip_dims and len(rms_errors) == 1:
            return rms_errors[0]
        else:
            return rms_errors
    
    def EvaluateNew(self, x, strip_dims = True):
        """
        Evaluate current fit(s) on a new set of x values.

        Args:
            strip_dims -- if True (default), return a single NumPy array rather than a list if a single data set is being fitted

        Notes:
            This method ignores the fitting range, if specified.
        """
        
        # Evaluate current fits with supplied x values.
        
        fit_eval = [
            func.Evaluate(x) for func in self._func_set
            ]
        
        # If strip_dims is set and only a single data set is being fit, return a single NumPy array.
        
        if strip_dims and len(fit_eval) == 1:
            return fit_eval[0]
        else:        
            return fit_eval
    
    def GetParamsList(self, strip_dims = True):
        """
        Collect a set of parameter lists for each function being fitted.

        Args:
            strip_dims -- if True (default), return a single list of parameters if a single data set is being fitted
        """
        
        params_list = [
            func.GetParamsList() for func in self._func_set
            ]
        
        # If strip_dims is set and only a single data set is being fit, return a single list of parameters.
        
        if strip_dims and len(params_list) == 1:
            return params_list[0]
        else:
            return params_list
