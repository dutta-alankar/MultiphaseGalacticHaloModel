# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:55:40 2023

@author: alankar
"""
import numpy as np
import os
import matplotlib.pyplot as plt
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
matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = 10.0
matplotlib.rcParams["xtick.minor.pad"] = 10.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 24.0
matplotlib.rcParams["ytick.labelsize"] = 24.0
matplotlib.rcParams["axes.titlesize"] = 24.0
matplotlib.rcParams["axes.labelsize"] = 28.0
matplotlib.rcParams["axes.labelpad"] = 8.0
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

# See illustris-analysis/diff-emm-plot_data.py in https://github.com/dutta-alankar/cooling-flow-model.git
tng50 = np.loadtxt("./Illustris-TNG50-1/tng50-pdf-data.txt")

plt.figure(figsize=(13, 10))
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 1] / np.log(10),
    color="darkgoldenrod",
    linewidth=3,
    linestyle="--",
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 2] / np.log(10),
    color="yellowgreen",
    linewidth=3,
    linestyle="--",
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 3] / np.log(10),
    color="slateblue",
    linewidth=3,
    linestyle="--",
)
'''
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 4] / np.log(10),
    color="magenta",
    linewidth=3,
    linestyle="--",
)
'''
phases_data = np.load("./figures/3PhasesPdf.npy")

Temperature = 10.0 ** phases_data[:, 0]
V_pdf = phases_data[:, 1]
M_pdf_m = phases_data[:, 2]
L_pdf_m = phases_data[:, 3]
V_pdf_c = phases_data[:, 4]
V_pdf_w = phases_data[:, 5]
V_pdf_h = phases_data[:, 6]
M_pdf_c = phases_data[:, 7]
M_pdf_w = phases_data[:, 8]
M_pdf_h = phases_data[:, 9]
cold_lum = phases_data[:, 10]
warm_lum = phases_data[:, 11]
hot_lum = phases_data[:, 12]

plt.plot(Temperature, V_pdf, color="darkgoldenrod", label="volume PDF", linewidth=6)
plt.plot(Temperature, M_pdf_m, color="yellowgreen", label="mass PDF", linewidth=6)
plt.plot(Temperature, L_pdf_m, color="slateblue", label="luminosity PDF", linewidth=6)

plt.plot(Temperature, V_pdf_c, color="darkgoldenrod", linewidth=6)
plt.plot(Temperature, V_pdf_w, color="darkgoldenrod", linewidth=6)
plt.plot(Temperature, V_pdf_h, color="darkgoldenrod", linewidth=6)
plt.plot(Temperature, M_pdf_h, color="yellowgreen", linewidth=6)
plt.plot(Temperature, M_pdf_w, color="yellowgreen", linewidth=6)
plt.plot(Temperature, M_pdf_c, color="yellowgreen", linewidth=6)
plt.plot(Temperature, hot_lum, color="slateblue", linewidth=6)
plt.plot(Temperature, cold_lum, color="slateblue", linewidth=6)
plt.plot(Temperature, warm_lum, color="slateblue", linewidth=6)


plt.xscale("log")
plt.yscale("log")
plt.ylim(10.0**-3.1, 10**0.2)
plt.xlim(10.0**3.99, 10.0**6.4)
plt.xlabel(r"Temperature [$K$]", size=28)
plt.ylabel(r"$T \mathscr{P}(T)$", size=28)
leg = plt.legend(loc="best", ncol=1, fancybox=True, fontsize=24, framealpha=0.5)
plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
plt.tight_layout()
# plt.grid()
# leg.set_title("Three phase PDF compared with a typical Illustris TNG50 Halo PDF", prop={'size':20})
os.system("mkdir -p ./figures")
plt.savefig("./figures/3-phases-pdf.png", transparent=False)
plt.show()
plt.close()
