#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:18:49 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import time
import os
import pickle
from typing import Union, Tuple
from scipy.optimize import minimize
from multiprocessing import Pool
import sys
sys.path.append("..")
from misc.coolLambda import cooling_approx
import importlib
volpdf = importlib.import_module("threePhases-param_find")
# globals().update(vars(add_volpdf))

threePhases = volpdf.threePhases

def mass_vol_pdf(
        params: Union[list, np.ndarray], logTemperature: Union[list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    # ----------- Taken from Vol PDF fit ---------------------------------
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

    T = 10.0**np.array(logTemperature)
    
    T_u = 10.0 ** params[0]
    T_h = T_u
    T_w = 10.0 ** params[1]
    T_c = 10.0 ** params[2]

    f_Vh = params[3]
    f_Vw = params[4]
    f_Vc = 1 - (f_Vh + f_Vw)

    sig_h = params[5]
    sig_w = params[6]
    sig_c = params[7]
    
    sig_u = sig_h
    T_u_M = T_u * np.exp(-(sig_u**2) / 2)
    '''
    f_Vh = 0.935
    f_Vw = 0.064
    f_Vc = 1 - (f_Vh + f_Vw)

    sig_u = 0.476
    T_u = 10.0**5.764
    T_u_M = T_u * np.exp(-(sig_u**2) / 2)

    T_h = T_u  # 10.**5.8
    T_w = 10**5.290
    T_c = 10.0**4.092

    sig_h = 0.476
    sig_w = 1.138
    sig_c = 0.177
    '''
    # mid-way between isochoric=1 and isobaric=0;
    # consistent prescription (beta is for across phases and del is within a phase)
    x = np.log(T / T_u) 
    x_h = np.log(T_h / T_u)
    x_w = np.log(T_w / T_u)
    x_c = np.log(T_c / T_u)
    # ----------- Taken from Vol PDF fit ---------------------------------
    
    del_h = params[8] 
    del_w = params[9] 
    del_c = params[10] 

    c_h = params[11]
    c_w = params[12]
    c_c = params[13] 

    beta_h = (c_h + del_h * (x_h + sig_u**2 / 2 + del_h * sig_h**2 / 2) )/(x_h + sig_u**2 / 2 + sig_h**2 * (del_h - 0.5))
    
    beta_w = (c_w +
        del_w
        * (x_w + sig_u**2 / 2 + del_w * sig_w**2 / 2) )/(x_w + sig_u**2 / 2 + sig_w**2 * (del_w - 0.5))
    
    beta_c = (c_c + 
        del_c
        * (x_c + sig_u**2 / 2 + del_c * sig_c**2 / 2) )/(x_c + sig_u**2 / 2 + sig_c**2 * (del_c - 0.5))
    

    # print ("beta_h, beta_w, beta_c = ", beta_h, beta_w, beta_c)

    T_h_M = T_h * np.exp((del_h - 0.5) * sig_h * sig_h)
    T_w_M = T_w * np.exp((del_w - 0.5) * sig_w * sig_w)
    T_c_M = T_c * np.exp((del_c - 0.5) * sig_c * sig_c)

    f_Mh = f_Vh * (T_h_M / T_u_M) ** (beta_h - 1)
    f_Mw = f_Vw * (T_w_M / T_u_M) ** (beta_w - 1)
    f_Mc = f_Vc * (T_c_M / T_u_M) ** (beta_c - 1)

    rho = f_Mh + f_Mw + f_Mc
    f_Mh = f_Mh / rho
    f_Mw = f_Mw / rho
    f_Mc = f_Mc / rho

    # print("volume fractions, hot, warm, cold:", f_Vh, f_Vw, f_Vc)
    # print("mass fractions, hot, warm, cold:", f_Mh, f_Mw, f_Mc)

    rho_av_u = 1.0
    rho_av_h = rho_av_u * (T_h_M / T_u_M) ** (beta_h - 1)
    rho_av_w = rho_av_u * (T_w_M / T_u_M) ** (beta_w - 1)
    rho_av_c = rho_av_u * (T_c_M / T_u_M) ** (beta_c - 1)
    rho_av = f_Vh * rho_av_h + f_Vw * rho_av_w + f_Vc * rho_av_c

    M_pdf_m = (
        f_Vh * np.exp(c_h)  
        * np.exp((del_h - 1) * (x_h + (del_h - 1) * sig_h**2 / 2 + sig_u**2 / 2))
        * N_pdf(x, x_h - (1 - del_h) * sig_h * sig_h, sig_h)
    )
    M_pdf_m += (
        f_Vw * np.exp(c_w)
        * np.exp((del_w - 1) * (x_w + (del_w - 1) * sig_w**2 / 2 + sig_u**2 / 2))
        * N_pdf(x, x_w - (1 - del_w) * sig_w * sig_w, sig_w)
    )
    M_pdf_m += (
        f_Vc * np.exp(c_c)
        * np.exp((del_c - 1) * (x_c + (del_c - 1) * sig_c**2 / 2 + sig_u**2 / 2))
        * N_pdf(x, x_c - (1 - del_c) * sig_c * sig_c, sig_c)
    )
    M_pdf_m *= rho_av_u / rho_av
    
    V_pdf_m = (
        f_Vh * N_pdf(x, x_h, sig_h)
        + f_Vw * N_pdf(x, x_w, sig_w)
        + f_Vc * N_pdf(x, x_c, sig_c)
    )
    
    return (np.log10(V_pdf_m), np.log10(M_pdf_m))

def log_likelihood(
    params: Union[list, np.ndarray], x_data: np.ndarray, y_data: np.ndarray
) -> np.ndarray:
    # x_data: logT, y_data: logMpdf, logLpdf
    model = mass_vol_pdf(params, x_data) # 0: mass_pdf, 1: lum_pdf
    # yerr = 1./np.abs(np.log10(np.abs(y_data - model)))
    # lsq = np.log(np.product(normal(y_data, model, yerr)))
    lsq = -0.5 * ( np.sum((y_data[0,:] - model[0]) ** 2) 
                  + 
                  np.sum((y_data[1,:] - model[1]) ** 2) )
    return lsq

params_limit =  [(5.3, 6.0), # T_h
                 (4.9, 5.6),  # T_w
                 (3.99, 4.3),  # T_c
                 (0.91, 0.97),  # fv_h
                 (0.001, 0.08),  # fv_w
                 (0.1, 0.8), # sig_h
                 (0.1, 3.0), # sig_w
                 (0.01, 0.5), # sig_c
                 (1.0, 5.0), # del h
                 (-0.2, 0.01),  # del w
                 (-12.0, -1.0),  # del c
                 (1.0, 30.0),  # C h
                 (1.0, 30.0),  # C w
                 (-30.0, 0.0)] # C c

params_prior =  [(5.70, 2.0), # T_h
                 (5.1, 1.0),  # T_w
                 (4.10, 0.1),  # T_c
                 (0.946, 0.5),  # fv_h
                 (0.053, 0.5),  # fv_w
                 (0.45, 0.1),  # sig_h
                 (1.10, 1.0),  # sig_w
                 (0.1, 0.1), # sig_c
                 (2.5, 1.0), # del h
                 (-0.01, 0.1),  # del w
                 (-7.5, 2.0),  # del c
                 (15.0, 2.0),  # C h
                 (14.0, 2.0),  # C w
                 (-13.0, 2.0)] # C c

def log_prior(params):
    lnnormal = lambda x, mu, sig: -(
        np.log(np.sqrt(2 * np.pi) * sig) + ((x - mu) / (np.sqrt(2) * sig)) ** 2
    )
    
    lp = np.sum(
          [lnnormal(params[i], params_prior[i][0], params_prior[i][1]) for i in range(len(params_prior))]
    )
   
    for i in range(len(params_limit)):
        condition = params_limit[i][0] < params[i] < params_limit[i][1]

    if condition:
        return lp
    else:
        return -np.inf


def log_probability(params, x_data, y_data):
    lp = log_prior(params)
    ll = log_likelihood(params, x_data, y_data)
    if not (np.isfinite(lp)) or np.sum(np.isnan(ll)) > 0:
        return -np.inf
    return lp + ll


if __name__ == "__main__":
    multitask = False
    os.makedirs("mkdir -p ./figures/", exist_ok=True)

    # See illustris-analysis/diff-emm-plot_data.py in https://github.com/dutta-alankar/cooling-flow-model.git
    tng50 = np.loadtxt("./Illustris-TNG50-1/tng50-pdf-data.txt")

    Temperature_data = tng50[:, 0]
    # We deliberately neglect the super virial phase
    clip = np.logical_and(Temperature_data > 3.99, Temperature_data <= 6.4)
    Temperature_data = Temperature_data[clip]
    pdf_data = np.zeros((2, Temperature_data.shape[0]), dtype=np.float64)
    
    for i in range(2):
        pdf = np.log10(
            np.piecewise(
                tng50[:, i+1],
                [
                    tng50[:, i+1] > 0.0,
                ],
                [lambda x: x, lambda x: 0.1 * np.min(tng50[:, i+1][tng50[:, i+1] > 0.0])],
            )
            / np.log(10)
        )    
        pdf_data[i,:] = pdf[clip]

    # Initial guess
    
    sig_u = 0.45
    T_u = 5.7

    f_Vh = 0.947
    f_Vw = 0.052
    f_Vc = 1.0 - (f_Vh + f_Vw)

    sig_h = 0.45
    sig_w = 1.10
    sig_c = 0.11

    T_h = T_u
    T_w = 5.20
    T_c = 4.10
    
    del_h = 2.5
    del_w = -0.01
    del_c = -7.5

    c_h = 15.0
    c_w = 15.0
    c_c = -12.0
    
    steps = 1000000
    random_factor = 1.0e-6

    params = np.array([T_h, T_w, T_c, f_Vh, f_Vw, sig_h, sig_w, sig_c, del_h, del_w, del_c, c_h, c_w, c_c])
    
    for i in range(len(params_limit)):
        condition = params_limit[i][0] < params[i] < params_limit[i][1]

    np.random.seed(int(time.time()))
    nll = lambda *args: -log_likelihood(*args)
    initial = params + random_factor * np.random.randn(params.shape[0])
    cons = ({"type": "ineq", "fun": lambda x: 1.0 - x[3] - x[4]},)
    bnds = params_limit
    
    soln = minimize(
        nll,
        initial,
        args=(Temperature_data, pdf_data),
        tol=1e-8,
        bounds=bnds,
        constraints=cons,
        options={"disp": True},
    )
    params_opt = soln.x

    print(
        "Initial likelihood value (neg): %.2e"
        % (-log_likelihood(params, Temperature_data, pdf_data))
    )
    print(
        "Final likelihood value (neg): %.2e"
        % (-log_likelihood(params_opt, Temperature_data, pdf_data))
    )
    print("Maximum likelihood estimates: ")
    print(params_opt)

    np.save("./figures/params_sol-mass_vol.npy", params_opt)

    nwalkers = 32
    ndim = params.shape[0]

    pos = params + random_factor * np.random.randn(nwalkers, ndim)
   

    if multitask:
        sampler = None
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                pool=pool,
                args=(Temperature_data, pdf_data),
            )
            sampler.run_mcmc(pos, steps, progress=True)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(Temperature_data, pdf_data),
    )
    sampler.run_mcmc(pos, steps, progress=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 18), sharex=True)
    samples = sampler.get_chain()
    labels = [
        r"$\log_{10}(T_h [K])$",
        r"$\log_{10}(T_w [K])$",
        r"$\log_{10}(T_c [K])$",
        r"$f_{V,h}$",
        r"$f_{V,w}$",
        r"$\sigma_{h}$",
        r"$\sigma_{w}$",
        r"$\sigma_{c}$",
        r"$\delta_h$",
        r"$\delta_w$",
        r"$\delta_c$",
        r"$c_h$",
        r"$c_w$",
        r"$c_c$",
    ]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], size=18)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number", size=18)
    plt.tight_layout()
    plt.savefig("./figures/emcee-walkers-mass_vol.png", transparent=False)
    plt.show()

    tau = sampler.get_autocorr_time()
    # print(tau)
    
    flat_samples = sampler.get_chain(
        discard=int(5 * np.ceil(np.max(tau))),
        thin=int(np.ceil(np.max(tau) / 2)),
        flat=True,
    )
    
    with open("./figures/corner_data-mass_vol.pickle", "wb") as f:
        data = {
            "flat_samples": flat_samples,
            "initial_guess": params,
        }
        # print(list(data.keys()))
        pickle.dump(data, f)

    fig = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.20, 0.50, 0.80],
        show_titles=True,
        title_kwargs={"fontsize": 16},
        label_kwargs={"fontsize": 16},
        truths=params,
        title_fmt=".3f",
    )
    plt.tight_layout()
    plt.savefig("./figures/emcee-params-mass_vol.png", transparent=False)
    plt.show()
    print("Initial guess: ", params)
