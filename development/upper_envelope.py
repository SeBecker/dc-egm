import numpy as np
from collections import namedtuple
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from copy import *
from numpy.matlib import * 
from scipy.optimize import *
import scipy.interpolate as scin



def interpolate(xx, obj, one=False):   
    """Interpolation function"""
    if not one:
        interpolation = scin.interp1d(obj[0], obj[1], bounds_error=False, fill_value="extrapolate")
        container = interpolation(xx)
        extrapolate = [True if (i>max(obj[0])) |(i<min(obj[0])) else False for i in xx]
    else:
        container = []
        extrapolate = []
        
        for poly  in obj:
            interpolation = scin.interp1d(poly[0], poly[1], bounds_error=False, fill_value="extrapolate")
            container += [interpolation(xx)]
            extrapolate += [np.array([True if (i>max(poly[0])) |(i<min(poly[0])) else False for i in xx])]
    return container, extrapolate


def chop(obj, j, repeat=None):
    """This function separates the grid into 1,..,j and j+1,...N parts."""
    for k in range(1):
        if j > len(obj[k]):
            j = len(obj[k])
        part1 = np.stack([obj[0][:j+1], obj[1][:j+1]])
        
        if repeat != None:
            # If repeat == True the boundary points are included in both arrays
            if repeat:
                part2 = np.stack([obj[0][j:], obj[1][j:]])
            else:
                part2 = np.stack([obj[0][j+1:], obj[1][j+1:]])
        if repeat is None:
            part2 = np.array([])

                
    return part1, part2