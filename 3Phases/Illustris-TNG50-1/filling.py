# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:49:30 2023

@author: alankar

Usage: python filling.py <halo_ID>
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
    file_hdf = h5py.File(f"halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

bins = 500

Group_Pos = np.array(file_hdf["Group_Pos"])
Group_r200 = np.array(file_hdf["Group_r200"])

position = np.array(file_hdf["/Coordinates"])
distance = np.sqrt(np.sum(np.array([(position[:,i]-Group_Pos[i])**2 for i in range(position.shape[1])]).T, axis=1))

condition = np.logical_and(np.log10(file_hdf["/Temperature"]) >= 3.99, np.log10(file_hdf["/Temperature"]) <= 7.0)
condition = np.logical_and(condition, np.logical_and(np.log10(file_hdf["/nH"]) >= -6.0, np.log10(file_hdf["/nH"]) <= -1.5))
condition = np.logical_and(condition, np.array(file_hdf["/SFR"])<=1.0e-06)
condition = np.logical_and(condition, distance<=1.1*Group_r200)

counts, xedges, yedges = np.histogram2d(
    x=np.log10(file_hdf["/nH"])[condition],
    y=np.log10(file_hdf["/Temperature"])[condition],
    weights=np.array(file_hdf["/Volume"])[condition],
    bins=(bins+1, bins),
    density=True,
)
xcenters = 0.5*(xedges[1:]+xedges[:-1])
ycenters = 0.5*(yedges[1:]+yedges[:-1])
xxcenters, yycenters = np.meshgrid(xcenters, ycenters)
print(xcenters.shape, ycenters.shape, counts.shape, xxcenters.shape)

vol_pdf, bin_edges = np.histogram(
    np.log10(file_hdf["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=np.array(file_hdf["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

vol_pdf = np.sum(counts.T, axis=1)*(xedges[1]-xedges[0])
print(np.trapz(vol_pdf, ycenters))
plt.semilogy(ycenters, vol_pdf, label="volume")

counts, xedges, yedges = np.histogram2d(
    x=np.log10(file_hdf["/nH"])[condition],
    y=np.log10(file_hdf["/Temperature"])[condition],
    weights=np.array(file_hdf["/Density"])[condition] * np.array(file_hdf["/Volume"])[condition],
    bins=(bins+1, bins),
    density=True,
)
xcenters = 0.5*(xedges[1:]+xedges[:-1])
ycenters = 0.5*(yedges[1:]+yedges[:-1])
xxcenters, yycenters = np.meshgrid(xcenters, ycenters)

mass_pdf, bin_edges = np.histogram(
    np.log10(file_hdf["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=np.array(file_hdf["/Density"])[condition] * np.array(file_hdf["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

mass_pdf = np.sum(counts.T, axis=1)*(xedges[1]-xedges[0])
print(np.trapz(mass_pdf, ycenters))
plt.semilogy(ycenters, mass_pdf, label="mass")


counts, xedges, yedges = np.histogram2d(
    x=np.log10(file_hdf["/nH"])[condition],
    y=np.log10(file_hdf["/Temperature"])[condition],
    weights=-np.array(file_hdf["/nH"])[condition] ** 2
    * np.array(file_hdf["/Lambda"])[condition]
    * np.array(file_hdf["/Volume"])[condition],
    bins=(bins+1, bins),
    density=True,
)
xcenters = 0.5*(xedges[1:]+xedges[:-1])
ycenters = 0.5*(yedges[1:]+yedges[:-1])
xxcenters, yycenters = np.meshgrid(xcenters, ycenters)

emm_pdf, bin_edges = np.histogram(
    np.log10(file_hdf["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=-np.array(file_hdf["/nH"])[condition] ** 2
    * np.array(file_hdf["/Lambda"])[condition]
    * np.array(file_hdf["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

emm_pdf = np.sum(counts.T, axis=1)*(xedges[1]-xedges[0])
print(np.trapz(emm_pdf, centers))
plt.semilogy(ycenters, emm_pdf, label="emission")

den_pdf, bin_edges = np.histogram(
    np.log10(file_hdf["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=-np.array(file_hdf["/Density"])[condition],
)

plt.semilogy(centers, den_pdf, label="density")
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

print(np.trapz(den_pdf, centers))


plt.legend(loc="best")
plt.ylim(ymin=2e-4)
plt.xlim(xmin=3.9, xmax=6.5)
plt.xlabel(r"Temperature [$K$]")
plt.ylabel(r"$T \mathscr{P}(T)$")
os.system("mkdir -p ./figures")
plt.savefig(f"./figures/halo-fill_ID={halo_id}.png", transparent=False)
# plt.show()

np.savetxt(
    "tng50-pdf-data.txt",
    np.vstack((centers, vol_pdf, mass_pdf, emm_pdf, den_pdf)).T,
    header="log10(T[K])  Vol_PDF  Mass_PDF  Diff_emm  Den_pdf")

file_hdf.close()
