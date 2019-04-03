# FitLib/FunctionsLibrary/General.py


# ----------------
# Module Docstring
# ----------------

""" Common functions for fitting. """


# -------
# Imports
# -------

import numpy as np

from FitLib.Function import CreateFunction


# ---------
# Functions
# ---------

def Polynomial(x, *p):
    """ General polynomial function y = p[0] * x ^ N + p[1] * x ^ (N - 1) + ... + p[N] """
    
    return np.polyval(p, x)

def CreatePolynomial(p, p_fit = None, p_bounds = None):
    """ Return a Function object representing a Polynomial function. """
    
    return CreateFunction(
        Polynomial, p, p_fit = p_fit, p_bounds = p_bounds
        )
