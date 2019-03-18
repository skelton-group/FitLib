# FitLib/Function.py


# ----------------
# Module Docstring
# ----------------

""" Core representations of functions and function parameters used by the Fitter class in the Fitter module. """


# -------
# Imports
# -------

import warnings


# ---------------
# Parameter Class
# ---------------

class Parameter:
    """ Class to encapsulate a function parameter. """
    
    def __init__(self, val, label, fit_flag = True, bounds = None):
        """
        Class constructor.
        
        Args:
            val -- initial parameter value
            label -- text label for parameter
            fit_flag -- True if parameter should be optimised, False if it should be fixed (default: True)
            bounds -- optionally specify a (p_min, p_max) range for parameter; either p_min or p_max can be None (default: None)
        """
        
        # Store data.
        
        self._val = float(val)
        self._label = str(label)
        self._fit_flag = bool(fit_flag)
        
        # If supplied, store fitting parameter bounds as a (p_min, p_max) tuple.
        
        if bounds is not None:
            p_min, p_max = bounds
            
            if p_min is not None:
                p_min = float(p_min)
            
            if p_max is not None:
                p_max = float(p_max)
            
            self._bounds = (p_min, p_max)

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
    
    @property
    def Label(self):
        """ Get text label for parameter. """
        
        return self._label
    
    @property
    def FitFlag(self):
        """ Fit flag: True if parameter is to be optimised, False otherwise. """
        
        return self._fit_flag
    
    @property
    def Bounds(self):
        """ Parameter bounds: (p_min, p_max) tuple or None if no bounds set. """
        
        return self._bounds


# --------------
# Function Class
# --------------

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
        
        param_labels = []
        
        for param in params:
            if param.Label in param_labels:
                raise Exception("Error: All function parameters must have a unique label.")
            else:
                param_labels.append(param.Label)
        
        self._params = params
    
    def Evaluate(self, x):
        """ Evaluate the function with the current parameters over the supplied x values. """
        
        return self._func(x, *[param.Value for param in self._params])
    
    def GetFitParams(self):
        """ Get a list of the parameters with the fit flag set. """
        
        return [param for param in self._params if param.FitFlag]
    
    def GetFitBounds(self):
        """ Get a list of bounds for parameters with the fit flag set. """
        
        return [param.Bounds for param in self._params if param.FitFlag]
    
    def GetParamsDict(self):
        """ Return a { label : value } dictionary for all function parameters, including those with the fit flag unset. """
        
        return { param.Label : param.Value for param in self._params }


# ----------------
# Factory Function
# ----------------

def CreateFunction(func, p_init, p_labels, p_fit = None, p_bounds = None):
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
        p_labels = [label for label in p_labels]
        
        assert len(p_labels) == len(params)
        
        if p_fit is not None:
            p_fit = [fit for fit in p_fit]
            
            assert len(p_fit) == len(params)
        
        if p_bounds is not None:
            p_bounds = [bounds for bounds in p_bounds]
            
            assert len(p_bounds) == len(params)
        
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                p_init[i] = Parameter(
                    param, p_labels[i], fit_flag = p_fit[i] if p_fit is not None else True, bounds = p_bounds[i] if p_bounds is not None else None
                    )
    
    # Return a function object for the supplied function and parameters.
    
    return Function(func, p_init)
