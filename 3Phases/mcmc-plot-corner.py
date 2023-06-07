# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:49:59 2023

@author: alankar
"""
import matplotlib.pyplot as plt
import corner
import pickle

with open("./figures/corner_data.pickle", "rb") as f:
    data = pickle.load(f)

flat_samples = data["flat_samples"]
params = data["initial_guess"]

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

fig = corner.corner(
    flat_samples,
    labels=labels,
    quantiles=[0.50, 0.50, 0.90],
    show_titles=True,
    title_kwargs={"fontsize": 16},
    label_kwargs={"fontsize": 16},
    truths=params,
    title_fmt=".3f",
)
plt.tight_layout()
plt.savefig("./figures/emcee-params.png", transparent=False)
#plt.show()
