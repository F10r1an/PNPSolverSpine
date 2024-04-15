#!/usr/bin/python

"""
this is a collection of various mathematical funcitons that are 
not yet implemented in the mpfr-package
"""

from gmpy2 import mpfr
import gmpy2


from .constants import constants
# set preciosion of mpfr variables
gmpy2.get_context().precision = constants['precision']

def cube(x):
    """
    function to cube number of mpfr-type
    x: mpfr-type number
    return: x^3
    """
    return x * x * x
    
def d1r(grid_points, mpfr_list, i, method='fwd'):
    """
    grid_points: list of x-values
    mpfr_list: list of y-values
    i: postiosn index
    compute first derivative with respect to r at r_i
    method: fwd - forward difference
            bwd - backward difference
            cd - central differnce with full step size
    """
    if method == 'fwd':
        return (mpfr_list[i+1] - mpfr_list[i]) / (grid_points[i+1] - grid_points[i])
    if method == 'bwd':
        return (mpfr_list[i] - mpfr_list[i-1]) / (grid_points[i] - grid_points[i-1])
    if method == 'cd':
        return (mpfr_list[i+1] - mpfr_list[i-1]) / (grid_points[i+1] - grid_points[i-1])
        
def d2r(grid_points, mpfr_list, i): 
    """
    grid_points: list of x-values
    mpfr_list: list of y-values
    i: postiosn index
    compute second derivative with respect to r at r_i
    function can only be used for a grid with linear spacing
    """
    return (mpfr_list[i+1] + mpfr_list[i-1]- mpfr('2') *  mpfr_list[i]) / (grid_points[i+1] - grid_points[i]) / (grid_points[i] - grid_points[i-1])


def mpfr_abs(x):
    return gmpy2.sqrt(gmpy2.square(x))

def interval_size(interval):
    if gmpy2.is_finite(interval[0]) and gmpy2.is_finite(interval[1]):
        return mpfr_abs(interval[1]-interval[0])
    else:
        return mpfr('inf')

def min_resolution(r_min, r_max, max_interval_size,scale):
    """
    min number of grid points that have a last interval smaller than arg: res_min
    but use at least 100 grid points
    """
    to_nm = mpfr(1.e9)  # method works in nanometer regime
    R = (mpfr(r_max) - mpfr(r_min)) * to_nm
    isz = mpfr(max_interval_size) * to_nm
    
    
    N=R/(R+ mpfr('1') - gmpy2.exp((R-isz)*gmpy2.log(R+mpfr('1'))/R ))
    N = gmpy2.ceil(N)
    return N





