# FitLib/Convenience.py


# ----------------
# Module Docstring
# ----------------

""" Contains convenience functions. """


# -------
# Imports
# -------

import itertools

from FitLib.Fitter import Fitter
from FitLib.Function import CreateFunction


# ---------
# Functions
# ---------

def SweepInitialParameters(data, func, params_or_param_lists, param_labels = None, fit_range = None, param_fit_flags = None, param_bounds = None, print_status = False, **kwargs):
    """
    Perform a series of fits of func to data with all combinations of specified initial parameters.
    
    Args:
        data -- (x, y) specifying data to fit
        func -- function to fit
        params_or_param_lists -- list of initial function parameters, either a single parameter or list of parameters to sweep
        param_labels -- optional parameter labels (passed to FitLib.Function.CreateFunction)
        fit_range -- optional range of x values to fit over (passed to FitLib.Function.CreateFunction)
        param_fit_flags -- optional list of True/False values specifying which parameters to optimise (True) or fix (False) during fitting (default: None; passed to FitLib.Function.CreateFunction)
        param_bounds -- optional list of (min, max) bounds for parameters during fitting (default: None = no bounds; passed to FitLib.Function.CreateFunction)
        print_status -- print status messages during parameter sweep (default: False)
    
    Returns:
        A (p_opt, fit_results) tuple containing the best set of parameters obtained from the sweep and a list of results for each combination of initial parameters tested.
        fit_results is a list of (p_init, p_opt, fit_rms) tuples specifying the initial/optimised parameters and the optimised root-mean-square (RMS) fitting error.
        fit_results is sorted by the RMS error.
    
    Notes:
        Keyword arguments are passed to scipy.optimize.minimize() via the Fitter.Fit() method.
        This function only accepts a single set of data and fitting function.
    """
    
    # Build parameter lists.
    
    param_lists = []
    
    for param_or_param_list in params_or_param_lists:
        try:
            # Assume param_or_param_list is an iterable and convert it to a list of floats.
            
            param_lists.append(
                [float(param) for param in param_or_param_list]
                )
        except TypeError:
            # Wrap scalar value in a list.
            
            param_lists.append(
                [float(param_or_param_list)]
                )
    
    # If param_labels is not set, initialise it a sensible default.
    
    if param_labels is not None:
        param_labels = [str(label) for label in param_labels]
    else:
        param_labels = ["p{0}".format(i + 1) for i in range(0, len(param_lists))]
    
    # Perform fits with all combinations of parameters.
    
    fit_results = []
    
    for p_init in itertools.product(*param_lists):
        fit_func = CreateFunction(func, p_init, param_labels, param_fit_flags, param_bounds)
        
        fitter = Fitter(data, fit_func, fit_range)
        fitter.Fit(**kwargs)
        
        p_opt = [
            param.Value for param in fitter.GetParamsList()
            ]
        
        fit_rms = fitter.EvaluateFitError()
        
        fit_results.append(
            (p_init, p_opt, fit_rms)
            )
        
        if print_status:
            val, label = p_init[0], param_labels[0]
           
            message = "{0} = {1: >6.3e}".format(label, val)
            
            for val, label in zip(p_init[1:], param_labels[1:]):
                message += ", {0} = {1: >6.3e}".format(label, val)
            
            message += " -> "
            
            val, label = p_opt[0], param_labels[0]
            
            message += "{0} = {1: >6.3e}".format(label, val)
            
            for val, label in zip(p_opt[1:], param_labels[1:]):
                message += ", {0} = {1: >6.3e}".format(label, val)
            
            message += " (rms = {0:.3e})".format(fit_rms)
            
            print(message)
    
    # Order by RMS.
    
    fit_results.sort(key = lambda item : item[-1])
    
    # Return the optimal parameters plus the results from the full set of trial fits.
    
    _, p_opt, _ = fit_results[0]
    
    return (p_opt, fit_results)
