import numpy as np
import scipy
import matplotlib.pyplot as plt 

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
    fit_si = fit_peak(data, 500, 540, lorentzian, [1, 520, 5, 0, 0])
    fit_g = fit_peak(data, 1500, 1700, fano, [0.1, 1580, 10, -10, 0, 0])
    fit_2d = fit_peak(data, 2500, 2900, lorentzian, [0.1, 2680, 10, 0, 0])
    return {'fit_si': fit_si[0], 'fit_g': fit_g[0], 'fit_2d': fit_2d[0]}

test = analyse('sample_data.txt', '\t')

##############
# Estimate n #
##############

def fit_i_2d_i_g(data):
    i_g = data['fit_g'][0]
    i_2d = data['fit_2d'][0]
    i2d_ig = i_2d / i_g
    solution_p = scipy.optimize.fsolve(lambda n: i2d_ig_spline(n) - i2d_ig, -1)
    solution_n = scipy.optimize.fsolve(lambda n: i2d_ig_spline(n) - i2d_ig, 3)
    return solution_p*1e13, solution_n*1e13

def fit_fwhm_g(data):
    fwhm_g = 2*data['fit_g'][2]
    fwhm_si = 2*data['fit_si'][2]
    solution_p_ef = scipy.optimize.fsolve(lambda efermi: (fwhm_g_correction_slg_vectorised(efermi, 300, 0.13) - fwhm_g_correction_slg_vectorised(20, 300, 0.13)) - (fwhm_g - fwhm_si), -1)
    solution_n_ef = scipy.optimize.fsolve(lambda efermi: (fwhm_g_correction_slg_vectorised(efermi, 300, 0.13) - fwhm_g_correction_slg_vectorised(20, 300, 0.13)) - (fwhm_g - fwhm_si), 1)

    solution_p = scipy.optimize.fsolve(lambda n: e_fermi_slg(n) - solution_p_ef, -1e13)
    solution_n = scipy.optimize.fsolve(lambda n: e_fermi_slg(n) - solution_n_ef, 1e13)
    #solution_p = scipy.optimize.fsolve(lambda n: (fwhm_g_spline(n) - broadening_das) - (fwhm_g - fwhm_si), -1)
    #solution_n = scipy.optimize.fsolve(lambda n: (fwhm_g_spline(n) - broadening_das) - (fwhm_g - fwhm_si), 3)
    return solution_p, solution_n

###################
# Estimate strain #
###################

def predict_pos_g(data, n, p):
    expected_shift_p = pos_g_spline(n/1e13)
    expected_shift_n = pos_g_spline(p/1e13)
    #solution_p = scipy.optimize.fsolve(lambda n: (pos_g_correction_slg_vectorised(n, 300, 0.13) - pos_g_correction_slg_vectorised(20, 300, 0.13)) - (pos_g - pos_si), -1)
    #solution_n = scipy.optimize.fsolve(lambda n: (pos_g_correction_slg_vectorised(n, 300, 0.13) - pos_g_correction_slg_vectorised(20, 300, 0.13)) - (pos_g - pos_si), 3)
    return expected_shift_p, expected_shift_n

def get_strain(pos, n, p):
    diff_g_p = pos[0] - test['fit_g'][1]
    diff_g_n = pos[1] - test['fit_g'][1]
    strain_g_p = diff_g_p / 23 # from Mohiuddin for uniaxial
    strain_g_n = diff_g_n / 23 # from Mohiuddin for uniaxial
    return strain_g_p, strain_g_n

##########
# Script #
##########

density_iratio = fit_i_2d_i_g(test)

print('Intensity Ratio')
print(f"i_2d / i_g = {test['fit_2d'][0] / test['fit_g'][0]}")
print(f"n_p = {density_iratio[0]}")
print(f"n_n = {density_iratio[1]}")

density_width = fit_fwhm_g(test)

print('FWHM')
print(f"FWHM(G) = {2*test['fit_g'][2]}")
print(f"FWHM(Si) = {2*test['fit_si'][2]}")
print(f"n_p = {density_width[0]}")
print(f"n_n = {density_width[1]}")

avg_p = (density_iratio[0] + density_width[0])/2
avg_n = (density_iratio[1] + density_width[1])/2

strain = get_strain(predict_pos_g(test, avg_n, avg_p), avg_n, avg_p)

print('Strain')
print(f"Strain(G) [p] = {strain[0]}")
print(f"Strain(G) [n] = {strain[1]}")