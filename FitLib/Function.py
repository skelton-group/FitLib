# FitLib/Function.py


# ----------------
# Module Docstring
# ----------------

""" Core representations of functions and function parameters used by the Fitter class in the Fitter module. """


# -------
# Imports
# -------

import warnings

import numpy as np


# ---------------
# Parameter Class
# ---------------

class Parameter:
    """ Class to encapsulate a function parameter. """
    
    def __init__(self, val, bounds = None, fit_flag = True):
        """
        Class constructor.
        
        Args:
            val -- initial parameter value
            bounds -- optionally specify a (p_min, p_max) range for parameter; either of p_min or p_max can be None (default: None)
            fit_flag -- True if parameter should be optimised, False if it should be fixed (default: True)
        """
        
        # Store data.
        
        self._val = float(val)

        self.Bounds = bounds        
        self.FitFlag = fit_flag

    @property
    def Bounds(self):
        """ Get a (p_min, p_max) tuple specifying the lower and upper limits of the parameter value (None indicates that no bounds are set). """
        
        return self._bounds
    
    @Bounds.setter
    def Bounds(self, bounds):
        """ Set the (p_min, p_max) parameter bounds; p_min/p_max = None or bounds = None can be set to indicate that no bounds should be applied. """
        
        if bounds is not None:
            p_min, p_max = bounds
            
            if p_min is not None:
                p_min = float(p_min)
            
            if p_max is not None:
                p_max = float(p_max)
            
            self._bounds = (p_min, p_max)
        else:
            self._bounds = (None, None)
    
    @property
    def FitFlag(self):
        """ Get the flag indicating whether the parameter should be optimised. """
        
        return self._fit_flag
    
    @FitFlag.setter
    def FitFlag(self, fit_flag):
        """ Set the fit flag: True if the parameter is to be optimised, False otherwise. """
        
        self._fit_flag = bool(fit_flag)

    @property
    def Value(self):
        """ Get parameter value. """
        
        return self._val
    
    @Value.setter
    def Value(self, val):
        """ Set parameter value. """
        
        val = float(val)
        
        if self._bounds is not None:
            p_min, p_max = self._bounds
            
            if (p_min is not None and val < p_min) or (p_max is not None and val > p_max):
                warnings.warn("Parameter.Value is out of bounds.", RuntimeWarning)
        
        self._val = val
    
    def Clone(self):
        """ Return a copy of the current Parameter object. """
        
        return Parameter(self.Value, bounds = self.Bounds, fit_flag = self.FitFlag)

# ----------------
# Function Classes
# ----------------

class Function:
    """ Class to encapsulate a function. """
    
    def __init__(self, func, param_or_params):
        """
        Class constructor.
        
        Args:
            func -- callable func(x, *params)
            param_or_params -- Parameter object or iterable of Parameter objects encapsulating param(s) to be passed to func
        """
        
        # Store function and parameters.
        
        assert func is not None
        
        self._func = func
        
        assert param_or_params is not None
        
        params = None
        
        if isinstance(param_or_params, Parameter):
            params = [param_or_params]
        else:
            params = [param for param in param_or_params]
            
            for param in params:
                assert isinstance(param, Parameter)
        
        self._params = params
    
    def Evaluate(self, x):
        """ Evaluate the function with the current parameters over the supplied x values. """
        
        return np.asarray(
            self._func(x, *[param.Value for param in self._params]), dtype = np.float64
            )
    
    def GetParamsList(self, fit_only = False):
        """
        Get a list of function parameters.
        
        Args:
            fit_only -- if True, excludes parameters with FitFlag = False (default: False)
        """
        
        return [param for param in self._params if param.FitFlag or not fit_only]
    
    def CloneParamsList(self):
        """ Create a (deep) copy of the current parameter list. """
        
        return [param.Clone() for param in self._params]


class CompositeFunction(Function):
    """ Class to encapsulate a composite function built from multiple Function objects. """
    
    def __init__(self, funcs):
        """
        Class constructor.
        
        Args:
            funcs -- iterable of Function objects
        
        """
        assert funcs is not None
        
        funcs = [func for func in funcs]
        
        for func in funcs:
            assert isinstance(func, Function)
        
        self._funcs = funcs
    
    def Evaluate(self, x):
        """ Evaluate the function with the current parameters over the supplied x values. """
        
        return np.sum(self.EvaluateIndividual(x), axis = 0)
    
    def EvaluateIndividual(self, x):
        # Convert x to a NumPy array and check.
        
        x = np.asarray(x, dtype = np.float64)
        
        n_x, = np.shape(x)
        
        assert n_x > 0
        
        # Evaluate stored functions.
        
        return [func.Evaluate(x) for func in self._funcs]
    
    def GetParamsList(self, fit_only = False):
        """
        Get a list of function parameters.
        
        Args:
            fit_only -- if True, excludes parameters with FitFlag = False (default: False)
        """
        params_list = []
        
        for func in self._funcs:
            params_list += func.GetParamsList(fit_only = fit_only)
        
        return params_list


# ----------------
# Factory Function
# ----------------

def CreateFunction(func, p_init, p_fit = None, p_bounds = None):
    """ Factory function to simplify creating Function objects. """
    
    assert func is not None
    
    # Assume p_init is an iterable and convert to a list; if this fails, assume p_init is a single parameter and wrap in a list.
    
    params = None
    
    try:
        params = [param for param in p_init]
    except TypeError:
        params = [p_init]
    
    # Check whether we need to wrap any or all of the parameters in Parameter objects.
    
    wrap = False
    
    for param in params:
        if not isinstance(param, Parameter):
            wrap = True
            break

    # If required, build Parameter objects around initial parameters supplied as scalar values.
    # In this case, the label is taken from p_labels, and the optional fit_flag and bounds are taken from p_fit and p_bounds if supplied.
    
    if wrap:
        if p_fit is not None:
            p_fit = [fit for fit in p_fit]
            
            assert len(p_fit) == len(params)
        
        if p_bounds is not None:
            p_bounds = [bounds for bounds in p_bounds]
            
            assert len(p_bounds) == len(params)
        
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                params[i] = Parameter(
                    param, fit_flag = p_fit[i] if p_fit is not None else True, bounds = p_bounds[i] if p_bounds is not None else None
                    )
    
    # Return a function object for the supplied function and parameters.
    
    return Function(func, params)
