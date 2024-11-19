# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:29:39 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import time
import os
import pickle
from typing import Union
from scipy.optimize import minimize
from multiprocessing import Pool


def threePhases(
    params: Union[list, np.ndarray], logTemperature: Union[list, np.ndarray]
) -> np.ndarray:
    N_pdf = lambda x, mu, sig: (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(
        -(x - mu) * (x - mu) / (2.0 * sig * sig)
    )

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

    # mid-way between isochoric=1 and isobaric=0;
    # consistent prescription (beta is for across phases and del is within a phase)
    x = np.log(10.0 ** np.array(logTemperature) / T_u)
    T_h = T_u
    T_w = 10.0 ** params[1]
    T_c = 10.0 ** params[2]

    x_h = np.log(T_h / T_u)
    x_w = np.log(T_w / T_u)
    x_c = np.log(T_c / T_u)

    V_pdf = (
        f_Vh * N_pdf(x, x_h, sig_h)
        + f_Vw * N_pdf(x, x_w, sig_w)
        + f_Vc * N_pdf(x, x_c, sig_c)
    )

    return np.log10(V_pdf)


def log_likelihood(
    params: Union[list, np.ndarray], x_data: np.ndarray, y_data: np.ndarray
) -> np.ndarray:
    # x_data: logT, y_data: logVpdf
    model = threePhases(params, x_data)
    # yerr = 1./np.abs(np.log10(np.abs(y_data - model)))
    # lsq = np.log(np.product(normal(y_data, model, yerr)))
    lsq = -0.5 * np.sum((y_data - model) ** 2)
    return lsq

params_limit =  [(5.3, 6.0), # T_h
                 (4.9, 5.6),  # T_w
                 (3.99, 4.3),  # T_c
                 (0.91, 0.97),  # fv_h
                 (0.03, 0.08),  # fv_w
                 (0.1, 0.8), # sig_h
                 (0.1, 2.0), # sig_w
                 (0.01, 0.5)] # sig_c

params_prior = [(5.70, 2.0), # T_h
                 (5.1, 1.0),  # T_w
                 (4.10, 0.1),  # T_c
                 (0.946, 0.1),  # fv_h
                 (0.052, 0.01),  # fv_w
                 (0.40, 0.1),  # sig_h
                 (1.10, 0.5),  # sig_w
                 (0.1, 0.1)] # sig_c

def log_prior(params):
    lnnormal = lambda x, mu, sig: -(
        np.log(np.sqrt(2 * np.pi) * sig) + ((x - mu) / (np.sqrt(2) * sig)) ** 2
    )

    lp = np.sum(
          [lnnormal(params[i], params_prior[i][0], params_prior[i][1]) for i in range(len(params_prior))]
    )

    condition = params_limit[0][0] < params[0] < params_limit[0][1]
    for i in range(1, len(params_limit)):
        condition = condition and (params_limit[i][0] < params[i] < params_limit[i][1])
    
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
    V_pdf_data = np.log10(
        np.piecewise(
            tng50[:, 1],
            [
                tng50[:, 1] > 0.0,
            ],
            [lambda x: x, lambda x: 0.1 * np.min(tng50[:, 1][tng50[:, 1] > 0.0])],
        )
        / np.log(10)
    )
    # We deliberately neglect the super virial phase
    clip = np.logical_and(Temperature_data >= 3.99, Temperature_data <= 6.4)
    Temperature_data = Temperature_data[clip]
    V_pdf_data = V_pdf_data[clip]

    # Initial guess
    sig_u = 0.46
    T_u = 5.7

    f_Vh = 0.948
    f_Vw = 0.051
    f_Vc = 1 - (f_Vh + f_Vw)

    sig_h = 0.45
    sig_w = 1.10
    sig_c = 0.11

    T_h = T_u
    T_w = 5.20
    T_c = 4.10

    random_factor = 1.0e-6
    
    steps = 100000
    
    params = np.array([T_h, T_w, T_c, f_Vh, f_Vw, sig_h, sig_w, sig_c])

    np.random.seed(int(time.time()))
    nll = lambda *args: -log_likelihood(*args)
    #initial = params + random_factor * params * np.random.randn(params.shape[0])
    initial = params + random_factor * np.random.randn(params.shape[0])
    cons = ({"type": "ineq", "fun": lambda x: 1.0 - x[3] - x[4]},)
    bnds = params_limit
    soln = minimize(
        nll,
        initial,
        args=(Temperature_data, V_pdf_data),
        tol=1e-6,
        bounds=bnds,
        constraints=cons,
        options={"disp": True},
    )
    params_opt = soln.x

    print(
        "Initial likelihood value (neg): %.2e"
        % (-log_likelihood(params, Temperature_data, V_pdf_data))
    )
    print(
        "Final likelihood value (neg): %.2e"
        % (-log_likelihood(params_opt, Temperature_data, V_pdf_data))
    )
    print("Maximum likelihood estimates: ")
    print(params_opt)

    np.save("./figures/params_sol.npy", params_opt)

    nwalkers = 32
    ndim = params.shape[0]

    #pos = params + random_factor * params * np.random.randn(nwalkers, ndim)
    pos = params + random_factor* np.random.randn(nwalkers, ndim)

    if multitask:
        sampler = None
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                pool=pool,
                args=(Temperature_data, V_pdf_data, additional_data),
            )
            sampler.run_mcmc(pos, steps, progress=True)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(Temperature_data, V_pdf_data),
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
    ]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], size=18)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number", size=18)
    plt.tight_layout()
    plt.savefig("./figures/emcee-walkers.png", transparent=False)
    plt.show()

    tau = sampler.get_autocorr_time()
    # print(tau)

    flat_samples = sampler.get_chain(
        discard=int(5 * np.ceil(np.max(tau))),
        thin=int(np.ceil(np.max(tau) / 2)),
        flat=True,
    )
    with open("./figures/corner_data.pickle", "wb") as f:
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
    plt.savefig("./figures/emcee-params.png", transparent=False)
    plt.show()
    print("Initial guess: ", params)
