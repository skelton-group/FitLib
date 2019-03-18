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

def CreateFuncPolynomial(p, p_fit = False, p_bounds = None):
    """ Return a Function object representing a Polynomial function. """
    
    return CreateFunction(
        Polynomial, p, ["x^{0}".format(len(p) - i) for i in range(0, len(p))], p_fit = p_fit, p_bounds = p_bounds
        )
