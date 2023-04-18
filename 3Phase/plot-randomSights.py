# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:15:04 2023

@author: alankar
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.load("randomSight_e.8_i.6.npz")

impact = data["impact"]
col_dens = data["col_dens"]
rCGM = data["rCGM"]

for i in range(impact.shape[0]):
    plt.scatter(
        (impact[i] / 211) * np.ones(col_dens.shape[1]), col_dens[i, :], color="tab:blue"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("b/r200")
    plt.ylabel(r"Column Density [$cm^{-2}$]")
plt.grid()
plt.show()
