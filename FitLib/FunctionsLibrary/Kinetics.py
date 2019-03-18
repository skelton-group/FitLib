# FitLib/FunctionsLibrary/Kinetics.py


# ----------------
# Module Docstring
# ----------------

""" Common functions for fitting kinetic data. """


# -------
# Imports
# -------

import numpy as np

from FitLib.Function import CreateFunction


# ---------
# Functions
# ---------

def JMAK(x, a_0, a_inf, k, n):
    """ JMAK function: a(x) = a_inf + (a_0 - a_inf) * exp(-k * t ^ n) """
    return a_inf + (a_0 - a_inf) * np.exp(-1.0 * k * np.power(x, n))

def CreateJMAK(a_0, a_inf, k, n, p_fit = None):
    """ Return a Function object representing a JMAK equation. """
    
    return CreateFunction(JMAK, [a_0, a_inf, k, n], ["a_0", "a_inf", "k", "n"], p_fit = p_fit, p_bounds = [(0.0, 1.0), (0.0, 1.0), (None, None), (0.0, 4.0)])
