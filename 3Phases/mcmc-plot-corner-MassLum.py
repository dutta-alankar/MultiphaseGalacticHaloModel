#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 07:37:55 2023

@author: alankar
"""

import matplotlib.pyplot as plt
import corner
import pickle

with open("./figures/corner_data-mass_lum.pickle", "rb") as f:
    data = pickle.load(f)

flat_samples = data["flat_samples"]
params = data["initial_guess"]

labels = [
    r"$\delta_h$",
    r"$\delta_w$",
    r"$\delta_c$",
    r"$c_h$",
    r"$c_w$",
    r"$c_c$",
]

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
plt.savefig("./figures/emcee-params-mass_lum.png", transparent=False)
plt.show()
