import numpy as np
import scipy

from density_estimate.literature import *

#################
# Fitting funcs #
#################

def fano(x: float, amp_f: float, cen_f: float, wid_f: float, q: float, y_f: float, gradient: float) -> float:
    s = (x - cen_f) / wid_f
    return y_f + (gradient * x) + (amp_f * (1 + s / q) ** 2) / (1 + s ** 2)

def lorentzian(x: float | np.ndarray, amp_l: float, cen_l: float, wid_l: float, y_l: float, gradient: float) -> float | np.ndarray:
    return y_l + (gradient * x) + amp_l*wid_l**2/((x-cen_l)**2+wid_l**2)

###############
# Data import #
###############

def read_data(filename: str, delim: str) -> np.ndarray: # currently assumes csv, no header
    data = np.loadtxt(filename, delimiter=delim)
    data[:,1] = data[:,1] - np.min(data[:,1]) # rudimentary background subtraction
    data[:,1] = data[:,1] / np.max(data[:,1]) # and normalisation
    return data 

############
# Data fit #
############

def truncate_array(data, x_min: float, x_max: float):
    """
    Truncate the array to include only rows where the x values are within the specified bounds.
    Parameters:
        data (np.array): 2D array where the first column is x values and the second column is y values.
        x_min (float): Minimum x value to include.
        x_max (float): Maximum x value to include.
    Returns:
        np.array: Truncated array.
    """
    # Boolean mask for rows where x values are within the bounds
    mask = (data[:, 0] >= x_min) & (data[:, 0] <= x_max)
    # Apply the mask to truncate the array
    truncated_data = data[mask]
    return truncated_data

def fit_peak(data: np.ndarray, xmin: float, xmax: float, func: callable, guess: float) -> tuple:
    data_masked = truncate_array(data, xmin, xmax)
    popt, pcov = scipy.optimize.curve_fit(func, data_masked[:,0], data_masked[:,1], p0=guess)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def analyse(file: str, delim: str) -> dict:
    data = read_data(file, delim)
    fit_g = fit_peak(data, 1500, 1700, fano, [0.1, 1580, 10, -10, 0, 0])
    fit_2d = fit_peak(data, 2500, 2900, lorentzian, [0.1, 2680, 10, 0, 0])
    return {'fit_g': fit_g[0], 'fit_2d': fit_2d[0]}

##############
# Estimate n #
##############

