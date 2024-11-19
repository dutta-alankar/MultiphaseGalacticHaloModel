#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:57:00 2024

@author: alankar
"""
import numpy as np
import h5py
import sys
import matplotlib
import matplotlib.pyplot as plt

## Plot Styling
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
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
matplotlib.rcParams["figure.figsize"] = (13,10)
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

halo_id = 110
try:
    file = h5py.File(f"./halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

Group_Pos = np.array(file["Group_Pos"])
Group_r200 = np.array(file["Group_r200"])

position = np.array(file["/Coordinates"])
distance = np.sqrt(np.sum(np.array([(position[:,i]-Group_Pos[i])**2 for i in range(position.shape[1])]).T, axis=1))

condition = np.logical_and(np.log10(file["/Temperature"]) >= 3.99, np.log10(file["/Temperature"]) <= 7.0)
condition = np.logical_and(condition, np.logical_and(np.log10(file["/nH"]) >= -6.0, np.log10(file["/nH"]) <= -1.5))
condition = np.logical_and(condition, np.array(file["/SFR"])<=1.0e-06)
condition = np.logical_and(condition, distance<=Group_r200)

x_data = np.log10(file["/nH"])[condition]
y_data =np.log10(file["/Temperature"])[condition]
weights_data = np.log10(file["/Volume"])[condition]

bins = 200

counts2d, xedges2d, yedges2d = np.histogram2d(x_data, y_data, 
                        bins=(bins+1, bins),
                        density=True,
                        weights=weights_data)
xcenters2d = 0.5*(xedges2d[1:]+xedges2d[:-1])
ycenters2d = 0.5*(yedges2d[1:]+yedges2d[:-1])

pdf_marg = np.sum(counts2d.T, axis=0)*(yedges2d[1]-yedges2d[0])

counts, xedges = np.histogram(x_data, bins+1, density=True, weights=weights_data)
xcenters = 0.5*(xedges[1:]+xedges[:-1])

plt.semilogy(xcenters, counts, color="tab:blue", linestyle="-")
plt.semilogy(xcenters2d, pdf_marg, color="black", linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=5)

plt.xlim(-5.8, -1.6)
# plt.ylim(3.9, 6.4)
plt.show()
plt.close() 
