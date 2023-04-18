# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:31:45 2022

@author: prateek and alankar
"""
import sys

sys.path.append("..")
from misc.coolLambda import cooling_approx
import numpy as np
import os

N_pdf = lambda x, mu, sig: (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(
    -(x - mu) * (x - mu) / (2.0 * sig * sig)
)

Z0 = 1.0
ZrCGM = 0.3
p = Z0 / ZrCGM
metallicity = (
    1.5
    * (p - (p**2 - 1) * np.arcsin(1.0 / np.sqrt(p**2 - 1)))
    * Z0
    * np.sqrt(p**2 - 1)
)

T = np.logspace(3.0, 8.0, 1000)

# params = np.load('./figures/params_sol.npy')

# f_Vh = params[3]
# f_Vw = params[4]
# f_Vc = 1 - (f_Vh+f_Vw)

# sig_u = 0.7
# T_u = 10.**params[0]
# T_h = T_u
# T_w = 10**params[1]
# T_c = 10.**params[2]

# T_u_M = T_u*np.exp(-sig_u**2/2)

# sig_h = params[5]
# sig_w = params[6]
# sig_c = params[7]

f_Vh = 0.938
f_Vw = 0.060
f_Vc = 1 - (f_Vh + f_Vw)

sig_u = 0.495
T_u = 10.0**5.754
T_u_M = T_u * np.exp(-(sig_u**2) / 2)

T_h = T_u  # 10.**5.8
T_w = 10**5.269
T_c = 10.0**4.102

sig_h = 0.495
sig_w = 1.054
sig_c = 0.300

# T_h = 10**6.4
# T_w = 10**5.2
# T_c = 1.5e4

# sig_h = 0.9
# sig_w = 1.4
# sig_c = 0.3

# mid-way between isochoric=1 and isobaric=0;
# consistent prescription (beta is for across phases and del is within a phase)
x = np.log(T / T_u)
x_h = np.log(T_h / T_u)
x_w = np.log(T_w / T_u)
x_c = np.log(T_c / T_u)

V_pdf = (
    f_Vh * N_pdf(x, x_h, sig_h)
    + f_Vw * N_pdf(x, x_w, sig_w)
    + f_Vc * N_pdf(x, x_c, sig_c)
)
np.save(
    "parameters.npy",
    np.array([f_Vh, f_Vw, f_Vc, x_h, x_w, x_c, sig_h, sig_w, sig_c, T_u]),
)
# print('V_pdf norm', np.trapz(V_pdf/T, T))


del_h = 1.1
del_w = 1e-4
del_c = 2e-3

beta_h = (
    del_h
    * (x_h + sig_u**2 / 2 + del_h * sig_h**2 / 2)
    / (x_h + sig_u**2 / 2 + sig_h**2 * (del_h - 0.5))
)
beta_w = (
    del_w
    * (x_w + sig_u**2 / 2 + del_w * sig_w**2 / 2)
    / (x_w + sig_w**2 / 2 + sig_w**2 * (del_w - 0.5))
)
beta_c = (
    del_c
    * (x_c + sig_u**2 / 2 + del_c * sig_c**2 / 2)
    / (x_c + sig_c**2 / 2 + sig_c**2 * (del_c - 0.5))
)

# print ("beta_h, beta_w, beta_c = ", beta_h, beta_w, beta_c)

T_h_M = T_h * np.exp((del_h - 0.5) * sig_h * sig_h)
T_w_M = T_w * np.exp((del_w - 0.5) * sig_w * sig_w)
T_c_M = T_c * np.exp((del_c - 0.5) * sig_c * sig_c)

f_Mh = f_Vh * (T_h_M / T_u_M) ** (beta_h - 1)
f_Mw = f_Vw * (T_w_M / T_u_M) ** (beta_w - 1)
f_Mc = f_Vc * (T_c_M / T_u_M) ** (beta_c - 1)

Den = f_Mh + f_Mw + f_Mc
f_Mh = f_Mh / Den
f_Mw = f_Mw / Den
f_Mc = f_Mc / Den

# print("volume fractions, hot, warm, cold:", f_Vh, f_Vw, f_Vc)
# print("mass fractions, hot, warm, cold:", f_Mh, f_Mw, f_Mc)

V_pdf_m = V_pdf
rho_av_u = 1.0
rho_av_h = rho_av_u * (T_h_M / T_u_M) ** (beta_h - 1)
rho_av_w = rho_av_u * (T_w_M / T_u_M) ** (beta_w - 1)
rho_av_c = rho_av_u * (T_c_M / T_u_M) ** (beta_c - 1)
rho_av = f_Vh * rho_av_h + f_Vw * rho_av_w + f_Vc * rho_av_c

M_pdf_m = (
    f_Vh
    * np.exp((del_h - 1) * (x_h + (del_h - 1) * sig_h**2 / 2 + sig_u**2 / 2))
    * N_pdf(x, x_h - (1 - del_h) * sig_h * sig_h, sig_h)
)
M_pdf_m += (
    f_Vw
    * np.exp((del_w - 1) * (x_w + (del_w - 1) * sig_w**2 / 2 + sig_u**2 / 2))
    * N_pdf(x, x_w - (1 - del_w) * sig_w * sig_w, sig_w)
)
M_pdf_m += (
    f_Vc
    * np.exp((del_c - 1) * (x_c + (del_c - 1) * sig_c**2 / 2 + sig_u**2 / 2))
    * N_pdf(x, x_c - (1 - del_c) * sig_c * sig_c, sig_c)
)
M_pdf_m *= rho_av_u / rho_av

# using exact luminosity function
hot_lum = (
    np.exp((del_h - 1) * sig_u**2)
    * f_Vh
    * cooling_approx(np.exp(x) * T_u, metallicity)
    * np.exp(2 * (del_h - 1) * (x_h + (del_h - 1) * sig_h * sig_h))
    * N_pdf(x, x_h + 2 * (del_h - 1) * sig_h * sig_h, sig_h)
)
L_pdf_m = hot_lum
warm_lum = (
    np.exp((del_w - 1) * sig_u**2)
    * f_Vw
    * cooling_approx(np.exp(x) * T_u, metallicity)
    * np.exp(2 * (del_w - 1) * (x_w + (del_w - 1) * sig_w * sig_w))
    * N_pdf(x, x_w + 2 * (del_w - 1) * sig_w * sig_w, sig_w)
)
L_pdf_m += warm_lum
cold_lum = (
    np.exp((del_c - 1) * sig_u**2)
    * f_Vc
    * cooling_approx(np.exp(x) * T_u, metallicity)
    * np.exp(2 * (del_c - 1) * (x_c + (del_c - 1) * sig_c * sig_c))
    * N_pdf(x, x_c + 2 * (del_c - 1) * sig_c * sig_c, sig_c)
)
L_pdf_m += cold_lum
L_pdf_m /= np.trapz(L_pdf_m, x)

f_Lh = np.trapz(hot_lum, x)
f_Lw = np.trapz(warm_lum, x)
f_Lc = np.trapz(cold_lum, x)
Den = f_Lh + f_Lw + f_Lc
f_Lh = f_Lh / Den
f_Lw = f_Lw / Den
f_Lc = f_Lc / Den

os.system("mkdir -p ./figures")
np.save("./figures/3PhasePdf.npy", np.vstack((np.log10(T), V_pdf, M_pdf_m, L_pdf_m)).T)
