# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:55:40 2023

@author: alankar
"""
import numpy as np
import os
import matplotlib.pyplot as plt

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

phases_data = np.load("./figures/3PhasePdf.npy")

Temperature = 10.0 ** phases_data[:, 0]
V_pdf = phases_data[:, 1]
M_pdf_m = phases_data[:, 2]
L_pdf_m = phases_data[:, 3]

plt.plot(Temperature, V_pdf, color="darkgoldenrod", label="volume PDF", linewidth=6)
plt.plot(Temperature, M_pdf_m, color="yellowgreen", label="mass PDF", linewidth=6)
plt.plot(Temperature, L_pdf_m, color="slateblue", label="luminosity PDF", linewidth=6)

plt.xscale("log")
plt.yscale("log")
plt.ylim(10.0**-3.1, 3.0)
plt.xlim(10.0**3.8, 10.0**7.5)
plt.xlabel(r"Temperature [$K$]", size=28)
plt.ylabel(r"$T \mathscr{P}(T)$", size=28)
leg = plt.legend(loc="upper right", ncol=3, fancybox=True, fontsize=24, framealpha=0.5)
plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
plt.tight_layout()
# leg.set_title("Three phase PDF compared with a typical Illustris TNG50 Halo PDF", prop={'size':20})
os.system("mkdir -p ./figures")
plt.savefig("./figures/3-phase-pdf.png", transparent=False)
# plt.show()
plt.close()
