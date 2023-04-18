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
from typing import Union
from scipy.optimize import minimize
from multiprocessing import Pool


def threePhases(params: Union[list, np.ndarray], logTemperature: float) -> np.ndarray:
    N_pdf = lambda x, mu, sig: (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(
        -(x - mu) * (x - mu) / (2.0 * sig * sig)
    )

    f_Vh = params[3]  # 0.967
    f_Vw = params[4]  # 0.031
    f_Vc = 1 - (f_Vh + f_Vw)

    # print('params: ', f_Vh, f_Vw, f_Vc)

    T_u = 10.0 ** params[0]  # 10.**6.4

    sig_h = params[5]  # 0.52
    sig_w = params[6]  # 1.4
    sig_c = params[7]  # 0.4

    T_h = T_u
    T_w = 10.0 ** params[1]  # 10**5.5
    T_c = 10.0 ** params[2]  # 1.4e4

    # sig_c = 0.3

    # mid-way between isochoric=1 and isobaric=0;
    # consistent prescription (beta is for across phases and del is within a phase)
    x = np.log(10.0**logTemperature / T_u)
    T_h = T_u
    T_w = 10.0 ** params[1]  # 10**5.5
    T_c = 10.0 ** params[2]  # 1.4e4

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


def log_prior(params):
    lnnormal = lambda x, mu, sig: -np.log(np.sqrt(2 * np.pi) * sig) - (x - mu) * (
        x - mu
    ) / (2.0 * sig * sig)

    global Th_expect, Tw_expect, Tc_expect
    x_h = Th_expect  # in log10
    x_w = Tw_expect  # in log10
    x_c = Tc_expect  # in log10

    lp = np.sum(
        lnnormal(params[0], x_h, 0.2)
        + lnnormal(params[1], x_w, 0.6)
        + lnnormal(params[2], x_c, 0.007)
        + lnnormal(params[7], 0.3, 0.01)
    )

    f_Vh = params[3]  # 0.967
    f_Vw = params[4]  # 0.031
    # f_Vc = 1 - (f_Vh+f_Vw)

    # T_u = 10.**params[0] # 10.**6.4

    sig_h = params[5]  # 0.52
    sig_w = params[6]  # 1.4
    sig_c = params[7]  # 0.4

    # T_h = T_u
    # T_w = 10.**params[1] # 10**5.5
    # T_c = 10.**params[2] # 1.4e4

    condition = 0.90 < f_Vh < 0.98
    condition = condition and (0.01 < f_Vw < (1.0 - f_Vh))
    # condition = condition and (5.8 <= np.log10(T_u) <= 6.8)
    # condition = condition and (4.9 <= np.log10(T_w) <= 5.6)
    # condition = condition and (4.0 <= np.log10(T_c) <= 5.4)
    condition = condition and (0.2 <= sig_h <= 0.9)
    condition = condition and (0.4 <= sig_w <= 1.8)
    condition = condition and (0.2 <= sig_c <= 0.5)

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
    os.system("mkdir -p ./figures/")

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
    clip = Temperature_data <= 6.5
    Temperature_data = Temperature_data[clip]
    V_pdf_data = V_pdf_data[clip]

    # Initial guess
    sig_u = 0.52
    T_u = 5.75

    f_Vh = 0.960
    f_Vw = 0.04
    f_Vc = 1 - (f_Vh + f_Vw)

    sig_h = 0.504
    sig_w = 1.1
    sig_c = 0.31

    T_h = T_u
    T_w = 4.92
    T_c = 4.101

    Th_expect = T_h
    Tw_expect = T_w
    Tc_expect = T_c

    random_factor = 1.0e-3

    params = np.array([T_h, T_w, T_c, f_Vh, f_Vw, sig_h, sig_w, sig_c])

    np.random.seed(int(time.time()))
    nll = lambda *args: -log_likelihood(*args)
    initial = params + random_factor * params * np.random.randn(params.shape[0])
    cons = ({"type": "ineq", "fun": lambda x: 1.0 - x[3] - x[4]},)
    bnds = (
        (5.4, 6.8),
        (4.2, 5.4),
        (4.0, 4.2),
        (0.5, 1.0),
        (0, 0.5),
        (0.1, 2.0),
        (0.5, 4.0),
        (0.05, 2.0),
    )
    soln = minimize(
        nll,
        initial,
        args=(Temperature_data, V_pdf_data),
        tol=1e-8,
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

    pos = params + random_factor * params * np.random.randn(nwalkers, ndim)

    if multitask:
        sampler = None
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                pool=pool,
                args=(Temperature_data, V_pdf_data),
            )
            sampler.run_mcmc(pos, 1000000, progress=True)

    steps = 100000
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(Temperature_data, V_pdf_data)
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
    # plt.show()

    tau = sampler.get_autocorr_time()
    print(tau)

    flat_samples = sampler.get_chain(
        discard=int(5 * np.ceil(np.max(tau))),
        thin=int(np.ceil(np.max(tau) / 2)),
        flat=True,
    )
    np.save("./figures/corner_data.npy", flat_samples)
    fig = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        truths=params,  # params,
        title_fmt=".3f",
    )
    plt.tight_layout()
    plt.savefig("./figures/emcee-params.png", transparent=False)
    # plt.show()
    print("Initial guess: ", params)
