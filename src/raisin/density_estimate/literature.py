import numpy as np
from scipy.interpolate import splrep, BSpline # for smoothed interpolation
import pandas as pd
import os

################################################################
# A script which fits splines to the data of Das et al. (2008) #
# to be used in electron density calculations                  #
################################################################

# importing the data from Das

ratio_i2d_ig_data = pd.read_csv('src/raisin/density_estimate/data/i_2d_i_g_vs_n_das_expt.txt', delimiter='\t', header=None)
fwhm_g_data = pd.read_csv('src/raisin/density_estimate/data/fwhm_g_vs_n_das_expt.txt', delimiter='\t', header=None)

pos_g_data = pd.read_csv('src/raisin/density_estimate/data/pos_g_vs_n_das_expt.txt', delimiter='\t', header=None)
pos_2d_data = pd.read_csv('src/raisin/density_estimate/data/pos_2d_vs_n_das_expt.txt', delimiter='\t', header=None)

# fitting smoothed splines to the data

tck_ratio_i2d_ig = splrep(ratio_i2d_ig_data[0], ratio_i2d_ig_data[1], s = (len(ratio_i2d_ig_data[0]) - 2*np.sqrt(2*len(ratio_i2d_ig_data[0]))))
i2d_ig_spline = BSpline(*tck_ratio_i2d_ig)

tck_fwhm_g = splrep(fwhm_g_data[0], fwhm_g_data[1], s = (len(fwhm_g_data[0]) - 2*np.sqrt(2*len(fwhm_g_data[0]))))
fwhm_g_spline = BSpline(*tck_fwhm_g)

tck_pos_g = splrep(pos_g_data[0], pos_g_data[1], s = (len(pos_g_data[0]) - 2*np.sqrt(2*len(pos_g_data[0]))))
pos_g_spline = BSpline(*tck_pos_g)

tck_pos_2d = splrep(pos_2d_data[0], pos_2d_data[1], s = (len(pos_2d_data[0]) - 2*np.sqrt(2*len(pos_2d_data[0]))))
pos_2d_spline = BSpline(*tck_pos_2d)