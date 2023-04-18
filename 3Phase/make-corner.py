#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:49:59 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import corner

# Initial guess
sig_u = 0.7
T_u = 5.6

f_Vh = 0.93
f_Vw = 0.0636
f_Vc = 1 - (f_Vh+f_Vw)

sig_h = 0.504
sig_w = 1.1
sig_c = 0.31

T_h = T_u
T_w = 5.0
T_c = 4.1

params = np.array([T_h,T_w,T_c, f_Vh,f_Vw, sig_h,sig_w,sig_c])
labels = [r"$\log_{10}(T_h [K])$", r"$\log_{10}(T_w [K])$", r"$\log_{10}(T_c [K])$", 
          r"$f_{V,h}$", r"$f_{V,w}$", 
          r"$\sigma_{h}$", r"$\sigma_{w}$", r"$\sigma_{c}$"]

flat_samples = np.load('corner_data.npy')
fig = corner.corner(
    flat_samples, labels=labels, 
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize":12}, truths=params,
    title_fmt='.3f')
plt.tight_layout()
plt.savefig('emcee-params.png', transparent=True)
plt.show()