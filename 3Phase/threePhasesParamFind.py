#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:29:39 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from multiprocessing import Pool
import time

flag = True
knowledge = None

def threePhases(params, logTemperature):
    N_pdf = lambda x, mu, sig: (1./(np.sqrt(2*np.pi)*sig))*np.exp(-(x-mu)*(x-mu)/(2.*sig*sig))
    
    f_Vh = params[3] # 0.967
    f_Vw = params[4] # 0.031
    f_Vc = 1 - (f_Vh+f_Vw)
    
    # print('params: ', f_Vh, f_Vw, f_Vc)
    
    T_u = 10.**params[0] # 10.**6.4
    
    sig_h = params[5] # 0.52
    sig_w = params[6] # 1.4
    sig_c = params[7] # 0.4
    
    T_h = T_u
    T_w = 10.**params[1] # 10**5.5
    T_c = 10.**params[2] # 1.4e4 
    
    # sig_c = 0.3
    
    # mid-way between isochoric=1 and isobaric=0; 
    # consistent prescription (beta is for across phases and del is within a phase)
    x = np.log(10.**logTemperature/T_u)
    T_h = T_u
    T_w = 10.**params[1] # 10**5.5
    T_c = 10.**params[2] # 1.4e4 
    
    x_h = np.log(T_h/T_u)
    x_w = np.log(T_w/T_u)
    x_c = np.log(T_c/T_u)
    
    V_pdf = f_Vh*N_pdf(x,x_h,sig_h) + f_Vw*N_pdf(x,x_w,sig_w) + f_Vc*N_pdf(x,x_c,sig_c)
    
    return np.log10(V_pdf)

def log_likelihood(params, x_data, y_data): #x_data -> logT, y_data -> logVpdf
    model = threePhases(params, x_data)
    
    normal = lambda x, mu, sig: (1./(np.sqrt(2*np.pi)*sig))*np.exp(-(x-mu)*(x-mu)/(2.*sig*sig))
    x_h = np.log10(3e6)
    x_w = np.log10(2e5)
    x_c = np.log10(2e4)
    
    # importance = normal(x_data,x_h,0.1) + normal(x_data,x_w,0.2) + normal(x_data,x_c,0.3)
    # normalization = np.max(importance)
    # importance = importance/normalization
    
    # yerr = np.abs(np.log10(np.abs(yerr/np.max(yerr)*10.**y_data)))
    
    # global flag
    # if (flag): print(yerr)
    # flag = False
    
    # yerr = 1./np.abs(np.log10(np.abs(y_data - model)))
    # lsq = np.log(np.product(normal(y_data, model, yerr))) 
    lsq = -0.5 * np.sum((y_data - model) ** 2 )
    # print("T_h=%.2f T_w=%.2f T_c=%.2f -lsq=%.5f"%(params[0], params[1], params[2], -lsq))
    return lsq

def log_prior(params):
    lnnormal = lambda x, mu, sig: -np.log(np.sqrt(2*np.pi)*sig)-(x-mu)*(x-mu)/(2.*sig*sig)
    
    x_h = np.log10(3e6)
    x_w = np.log10(2e5)
    x_c = np.log10(1.5e4)
    lp = np.sum(lnnormal(params[0],x_h,0.1) + lnnormal(params[1],x_w,0.2) + lnnormal(params[2],x_c,0.1) + \
                lnnormal(params[7],0.3,0.01))
    
    f_Vh = params[3] # 0.967
    f_Vw = params[4] # 0.031
    f_Vc = 1 - (f_Vh+f_Vw)
    
    T_u = 10.**params[0] # 10.**6.4
    
    sig_h = params[5] # 0.52
    sig_w = params[6] # 1.4
    sig_c = params[7] # 0.4
    
    T_h = T_u
    T_w = 10.**params[1] # 10**5.5
    T_c = 10.**params[2] # 1.4e4 
    
    condition = 0.95 < f_Vh < 1.0
    condition = condition and (0.01 < f_Vw < (1.-f_Vh))
    # condition = condition and (5.8 <= np.log10(T_u) <= 6.8)
    # condition = condition and (4.9 <= np.log10(T_w) <= 5.6)
    # condition = condition and (4.0 <= np.log10(T_c) <= 5.4)
    condition = condition and (0.1 <= sig_h <= 2.0)
    condition = condition and (0.5 <= sig_w <= 2.0)
    # condition = condition and (0.1 <= sig_c <= 0.6)
    
    if (condition): return lp
    else: return -np.inf
    
def log_probability(params, x_data, y_data):
    lp = log_prior(params)
    ll = log_likelihood(params, x_data, y_data)
    if not (np.isfinite(lp)) or np.sum(np.isnan(ll))>0:
        return -np.inf
    return lp + ll


# See illustris-analysis/diff-emm-plot_data.py in https://github.com/dutta-alankar/cooling-flow-model.git
tng50 = np.loadtxt('tng50-pdf-data.txt')

Temperature_data = tng50[:,0]
V_pdf_data = np.log10(tng50[:,1]/np.log(10))

# Initial guess
f_Vh = 0.957
f_Vw = 0.038
f_Vc = 1 - (f_Vh+f_Vw)

sig_u = 0.7
T_u = 6.4

sig_h = 0.52
sig_w = 1.4
sig_c = 0.4

T_h = T_u
T_w = 5.2
T_c = 4.1 

random_factor = 1.e-3

params = np.array([T_h,T_w,T_c, f_Vh,f_Vw, sig_h,sig_w,sig_c])

np.random.seed(int(time.time()))
nll = lambda *args: -log_likelihood(*args)
initial = params + random_factor * params * np.random.randn(params.shape[0])
cons = ({'type': 'ineq', 'fun': lambda x: 1. - x[3] - x[4] },)
bnds = ((5.8, 6.8),(4.8, 5.8),(4.0, 5.4), (0.5, 1.0),(0, 0.5), (0.1, 2.0),(0.5, 4.0),(0.05, 2.0))
soln = minimize(nll, initial, args=(Temperature_data, V_pdf_data), tol=1e-8,
                bounds=bnds, constraints=cons, options={'disp': True})
params_opt = soln.x

print('Initial likelihood value (neg): %.2e'%(-log_likelihood(params, Temperature_data, V_pdf_data)) )
print('Final likelihood value (neg): %.2e'%(-log_likelihood(params_opt, Temperature_data, V_pdf_data)) )
print("Maximum likelihood estimates: ")
print(params_opt)

np.save('params_sol.npy', params_opt)

nwalkers = 32 
ndim = params.shape[0]

pos = params + random_factor * params_opt * np.random.randn(nwalkers, ndim)

# sampler = None
# with Pool() as pool:
#     sampler = emcee.EnsembleSampler(
#         nwalkers, ndim, log_probability, pool=pool, args=(Temperature_data, V_pdf_data))
#     sampler.run_mcmc(pos, 1000000, progress=True)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(Temperature_data, V_pdf_data))
sampler.run_mcmc(pos, 100000, progress=True)

fig, axes = plt.subplots(ndim, figsize=(10, 18), sharex=True)
samples = sampler.get_chain()
labels = [r"$\log_{10}(T_h [K])$", r"$\log_{10}(T_w [K])$", r"$\log_{10}(T_c [K])$", 
          r"$f_{V,h}$", r"$f_{V,w}$", 
          r"$\sigma_{h}$", r"$\sigma_{w}$", r"$\sigma_{c}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i], size=18)
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number", size=18)
plt.tight_layout()
plt.savefig('emcee-walkers.png', transparent=True)
plt.show()

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=int(5*np.ceil(np.max(tau))), thin=int(np.ceil(np.max(tau)/2)), flat=True)
np.save('corner_data.npy', flat_samples)
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
