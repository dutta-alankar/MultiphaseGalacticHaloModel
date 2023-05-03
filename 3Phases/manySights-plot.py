# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:15:04 2023

@author: alankar
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 0.4
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["lines.linewidth"] = 3.0
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = 6.0
matplotlib.rcParams["xtick.minor.pad"] = 6.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 24.0
matplotlib.rcParams["ytick.labelsize"] = 24.0
matplotlib.rcParams["axes.titlesize"] = 24.0
matplotlib.rcParams["axes.labelsize"] = 28.0
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

all_ncl = [100, 1000, 10000]
color = ["tab:red", "tab:blue", "tab:green"]

plt.figure(figsize=(13, 10))

store = []
for indx, ncl in enumerate(all_ncl[::-1]):
    element = "MgII"
    data = np.load(f"./figures/randomSight_e.{element}_{ncl}.npz")

    impact = data["impact"]
    col_dens = data["col_dens"]
    col_dens_avg = np.average(col_dens, axis=1)
    col_dens_med = np.median(col_dens, axis=1)
    rCGM = data["rCGM"]
    ncl = data["ncl"]

    for i in range(impact.shape[0]):
        plt.scatter(
            (impact[i] / rCGM) * np.ones(col_dens.shape[1]),
            col_dens[i, :],
            color=color[indx],
        )
    store.append([impact / rCGM, col_dens_avg, col_dens_med])
    plt.plot(impact / rCGM, col_dens_avg, color=color[indx])

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$b/R_{vir}$", size=28)
plt.ylabel(r"Column density $[cm^{-2}]$", size=28)
# leg = plt.legend(loc="upper right", ncol=3, fancybox=True, fontsize=24, framealpha=0.5)
plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
plt.tight_layout()
plt.savefig(f"./figures/randomSight_e.{element}_all.png")
# plt.show()

plt.figure(figsize=(13, 10))

for indx, ncl in enumerate(all_ncl[::-1]):
    plt.plot(store[indx][0], store[indx][1], linestyle="-", color=color[indx])
    plt.plot(store[indx][0], store[indx][2], linestyle=":", color=color[indx])

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$b/R_{vir}$", size=28)
plt.ylabel(r"Column density $[cm^{-2}]$", size=28)
# leg = plt.legend(loc="upper right", ncol=3, fancybox=True, fontsize=24, framealpha=0.5)
plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
plt.tight_layout()
plt.savefig(f"./figures/randomSight_e.{element}_all_lines.png")
