# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:49:30 2023

@author: alankar
"""

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

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
matplotlib.rcParams["legend.handlelength"] = 2
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

halo_id = int(sys.argv[1])
try:
    file = h5py.File(f"halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

vol_pdf, bin_edges = np.histogram(
    np.log10(file["/Temperature"]),
    bins=300,
    density=True,
    weights=np.array(file["/Volume"]),
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

plt.semilogy(centers, vol_pdf, label="volume")

mass_pdf, bin_edges = np.histogram(
    np.log10(file["/Temperature"]),
    bins=300,
    density=True,
    weights=np.array(file["/NumberDensity"]) * np.array(file["/Volume"]),
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
plt.semilogy(centers, mass_pdf, label="mass")

emm_pdf, bin_edges = np.histogram(
    np.log10(file["/Temperature"]),
    bins=300,
    density=True,
    weights=-np.array(file["/nH"]) ** 2
    * np.array(file["/Lambda"])
    * np.array(file["/Volume"]),
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
plt.semilogy(centers, emm_pdf, label="emission")

plt.legend(loc="best")
plt.ylim(ymin=2e-4)
plt.xlim(xmin=3.9, xmax=7.6)
plt.xlabel(r"Temperature [$K$]")
plt.ylabel(r"$T \mathscr{P}(T)$")
os.system("mkdir -p ./figures")
plt.savefig(f"./figures/halo-fill_ID={sys.argv[1]}.png", transparent=False)
# plt.show()

np.savetxt(
    "tng50-pdf-data.txt",
    np.vstack((centers, vol_pdf, mass_pdf, emm_pdf)).T,
    header="log10(T[K])			Vol_PDF		Mass_PDF		Diff_emm",
)

file.close()
