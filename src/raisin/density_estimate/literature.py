import numpy as np
from scipy.interpolate import splrep, BSpline # for smoothed interpolation
from scipy import integrate
import pandas as pd
import os


################################################################
# A script which fits splines to the data of Das et al. (2008) #
# to be used in electron density calculations                  #
################################################################

# importing the data from Das

ratio_i2d_ig_data = pd.read_csv('~/Documents/raisin/src/raisin/density_estimate/data/i_2d_i_g_vs_n_das_expt.txt', delimiter='\t', header=None)
fwhm_g_data = pd.read_csv('~/Documents/raisin/src/raisin/density_estimate/data/fwhm_g_vs_n_das_expt.txt', delimiter='\t', header=None)

pos_g_data = pd.read_csv('~/Documents/raisin/src/raisin/density_estimate/data/pos_g_vs_n_das_expt.txt', delimiter='\t', header=None)
pos_2d_data = pd.read_csv('~/Documents/raisin/src/raisin/density_estimate/data/pos_2d_vs_n_das_expt.txt', delimiter='\t', header=None)

broadening_das = 2.5 # an estimate, based on comparison with theory from Lazzeri & Mauri (2006)

# fitting smoothed splines to the data

tck_ratio_i2d_ig = splrep(ratio_i2d_ig_data[0], ratio_i2d_ig_data[1], s = 0.02 * (len(ratio_i2d_ig_data[0]) - 2*np.sqrt(2*len(ratio_i2d_ig_data[0]))))
i2d_ig_spline = BSpline(*tck_ratio_i2d_ig)

tck_fwhm_g = splrep(fwhm_g_data[0], fwhm_g_data[1], s = (len(fwhm_g_data[0]) - 2*np.sqrt(2*len(fwhm_g_data[0]))))
fwhm_g_spline = BSpline(*tck_fwhm_g)

tck_pos_g = splrep(pos_g_data[0], pos_g_data[1], s = (len(pos_g_data[0]) - 2*np.sqrt(2*len(pos_g_data[0]))))
pos_g_spline = BSpline(*tck_pos_g)

tck_pos_2d = splrep(pos_2d_data[0], pos_2d_data[1], s = (len(pos_2d_data[0]) - 2*np.sqrt(2*len(pos_2d_data[0]))))
pos_2d_spline = BSpline(*tck_pos_2d)

##########
# Theory #
##########

h = 4.135e-15 # eV s
c = 3e10 # cm/s
k_boltzmann = 8.617e-5 # eV / K

alpha_prime = 4.43e-3 # dimensionless
#alpha_prime = 5.09e-4 # λ/2π (as in the ppt)
# 2π α' is also called λ_Γ in literature
E_ph = 0.196 # eV, equivalent photon energy E = hωc

def f(epsilon, e_fermi, T): # defining the Fermi-Dirac distribution in the conventional way
    return (1 / (np.exp((epsilon - e_fermi) / (k_boltzmann * T)) + 1))

def pi_slg(e_fermi, T, delta): # the phonon self-energy for SLG. The 0.13i is an empirical broadening
    epsilon = np.linspace(-10,10,20000)
    return integrate.simpson(np.absolute(epsilon) * (f(epsilon, e_fermi, T) - f(-epsilon, e_fermi, T)) / (2*epsilon + complex(E_ph, delta)), epsilon) * alpha_prime

def pos_g_correction_slg(e_fermi, T, delta): # Pos(G) comes from the real part of the self-energy
    pi = pi_slg(e_fermi, T, delta)
    result = (pi - pi_slg(0, T, delta)).real / (h * c)
    return result

def fwhm_g_correction_slg(e_fermi, T, delta): # FWHM(G) comes from the imaginary part of the self-energy
    result = 2 * np.imag(pi_slg(e_fermi, T, delta) - pi_slg(0, T, delta)) / (h * c)
    return -result

pos_g_correction_slg_vectorised = np.vectorize(pos_g_correction_slg)
fwhm_g_correction_slg_vectorised = np.vectorize(fwhm_g_correction_slg)

##################
# EF(n) from TBA #
##################

def e_fermi_slg(n):
    hbar = h / (2*np.pi)
    if n < 0: 
        return - hbar * 1.1e8 * np.sqrt(np.pi * np.absolute(n))
    else:
        return hbar * 1.1e8 * np.sqrt(np.pi * np.absolute(n))
e_fermi_slg_vectorised = np.vectorize(e_fermi_slg)