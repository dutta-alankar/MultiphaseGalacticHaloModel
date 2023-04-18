#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:48:38 2023

@author: alankar
"""

from misc.coolLambda import cooling_approx
import numpy as np
# import matplotlib.pyplot as plt

N_pdf = lambda x, mu, sig: (1./(np.sqrt(2*np.pi)*sig))*np.exp(-(x-mu)*(x-mu)/(2.*sig*sig))

Z0   = 1.0
ZrCGM = 0.3
p = Z0/ZrCGM
metallicity = 1.5*(p-(p**2-1)*np.arcsin(1./np.sqrt(p**2-1)))*Z0*np.sqrt(p**2-1)

T = np.logspace(3., 8., 1000)

params = np.load('params_sol.npy')

f_Vh = 0.968
f_Vw = 0.026
f_Vc = 1 - (f_Vh+f_Vw)

sig_u = 0.7
T_u = 10.**6.370
T_u_M = T_u*np.exp(-sig_u**2/2)

sig_h = 0.665
sig_w = 1.162
sig_c = 0.299

T_h = T_u
T_w = 10**5.385
T_c = 10.**4.237
