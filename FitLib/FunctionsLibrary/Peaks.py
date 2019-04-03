# FitLib/FunctionsLibrary/Peaks.py


# ----------------
# Module Docstring
# ----------------

""" Common functions for peak fitting. """


# -------
# Imports
# -------

import math

import numpy as np

from FitLib.Function import CreateFunction


# ---------
# Constants
# ---------

""" Minimum bound to linewidths to avoid divide by zero errors. """

MinLinewidth = 1.0e-8


# ---------
# Functions
# ---------

def Gaussian(x, a, x_0, sigma):
    """
    Normalised Gaussian function G(x).
    Taken from mathworld.wolfram.com/GaussianFunction.html.
    """
    
    return (a / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-1.0 * (x - x_0) ** 2 / (2.0 * sigma ** 2))

def CreateGaussian(a, x_0, sigma, p_fit = None, force_positive = False):
    """ Return a Function object representing a Gaussian function. """
    
    return CreateFunction(
        Gaussian, [a, x_0, sigma], p_fit = p_fit, p_bounds = [(0.0 if force_positive else None, None), (None, None), (MinLinewidth, None)]
        )

def GaussianOverX(x, a, x_0, w):
    """
	"Gaussian over X" function for fitting pair-distribution function (PDF) data.
	Taken from: Granlund, Billinge and Duxbury, Acta. Cryst. A 71. 392-409 (2015), DOI: 10.1107/S2053273315005276.
	"""
    
    w_2 = w ** 2
    
    pre = a / math.sqrt((math.pi / (4.0 * math.log(2.0))) * w_2)
    exp = (-4.0 * math.log(2.0)) / w_2
    
    return (1.0 / x) * pre * np.exp(exp * (x - x_0) ** 2)

def CreateGaussianOverX(a, x_0, w, p_fit = None, force_positive = False):
    """ Return a Function object representing a "Gaussian over X" function. """
    
    return CreateFunction(
        GaussianOverX, [a, x_0, w], ["a", "x_0", "w"], p_fit = p_fit, p_bounds = [(0.0 if force_positive else None, None), (None, None), (MinLinewidth, None)]
        )
