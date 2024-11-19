# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:54:14 2024

@author: alankar
"""

import numpy as np
import h5py
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import pickle

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    taken from: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

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
matplotlib.rcParams["ytick.major.width"] = 2.0
matplotlib.rcParams["xtick.major.width"] = 2.0
matplotlib.rcParams["ytick.minor.width"] = 1.0
matplotlib.rcParams["xtick.minor.width"] = 1.0
matplotlib.rcParams["ytick.major.size"] = 10.0
matplotlib.rcParams["xtick.major.size"] = 10.0
matplotlib.rcParams["ytick.minor.size"] = 6.0
matplotlib.rcParams["xtick.minor.size"] = 6.0
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

halo_id = int(sys.argv[1])
try:
    file_hdf = h5py.File(f"halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

Group_Pos = np.array(file_hdf["Group_Pos"])
Group_r200 = np.array(file_hdf["Group_r200"])

position = np.array(file_hdf["/Coordinates"])
distance = np.sqrt(np.sum(np.array([(position[:,i]-Group_Pos[i])**2 for i in range(position.shape[1])]).T, axis=1))

condition = np.logical_and(np.log10(file_hdf["/Temperature"]) >= 3.99, np.log10(file_hdf["/Temperature"]) <= 7.0)
condition = np.logical_and(condition, np.logical_and(np.log10(file_hdf["/nH"]) >= -6.0, np.log10(file_hdf["/nH"]) <= -1.5))
condition = np.logical_and(condition, np.array(file_hdf["/SFR"])<=1.0e-06)
condition = np.logical_and(condition, distance<=Group_r200)

bins = 500

pc = plt.hist2d(
    x=np.log10(file_hdf["/Temperature"])[condition],
    y=np.log10(file_hdf["/ZbZSun"])[condition],
    weights=np.array(file_hdf["/Volume"])[condition],
    bins=(bins, bins),
    density=True,
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(vmin=2.0e-4, vmax=2.8),
    zorder=-0.5,
)
cbar = plt.colorbar(pad=0.01)
cbar.ax.set_ylabel(r"Volume weighted distribution")
cbar.ax.tick_params('both', length=10, width=2, which='major', direction='out')
cbar.ax.tick_params('both', length=6, width=0.8, which='minor', direction='out')
cbar.ax.tick_params(labelsize=22.0)

bins = 500
temperature = np.log10(file_hdf["/Temperature"])[condition]
metallicity = np.log10(file_hdf["/ZbZSun"])[condition]
weight = np.array(file_hdf["/Volume"])[condition]

temperature_bin_edges = np.log10(np.logspace(np.log10(file_hdf["/Temperature"]).min(), np.log10(file_hdf["/Temperature"]).max(), bins+1))
met_16, met_50, met_84 = np.zeros(bins, dtype=np.float64), np.zeros(bins, dtype=np.float64), np.zeros(bins, dtype=np.float64)

for this_bin in range(bins):
    select = np.logical_and(temperature>=temperature_bin_edges[this_bin], temperature<temperature_bin_edges[this_bin+1])
    if (len(metallicity[select]) == 0):
        met_16[this_bin], met_50[this_bin], met_84[this_bin] = np.nan, np.nan, np.nan
        continue
    met_16[this_bin], met_50[this_bin], met_84[this_bin] = weighted_quantile(metallicity[select], [0.16,0.5,0.84], weight[select])

temperature_bin_cens = 0.5*(temperature_bin_edges[1:]+temperature_bin_edges[:-1])
plt.plot(temperature_bin_cens, met_50, color="tab:red", linestyle="-", linewidth=4)
plt.plot(temperature_bin_cens, met_16, color="tab:red", linestyle="--", linewidth=2)
plt.plot(temperature_bin_cens, met_84, color="tab:red", linestyle="--", linewidth=2)

plt.hlines(np.log10([0.3, 0.5, 1.0]), 3.9, 7.0, color="k", linestyle="-.", linewidth=1)

plt.ylim(ymin=-2.48, ymax=0.9)
plt.xlim(xmin=3.99, xmax=6.99)
plt.xlabel(r"Gas Temperature $[log_{10}\ K]$")
plt.ylabel(r"Metallicity [$log_{10}(Z/Z_{\odot})$]")

plt.savefig("./figures/metallicity-trend.png", bbox_inches='tight', transparent=False)
np.savetxt("./met-trend.txt", np.vstack( (temperature_bin_cens, met_16, met_50, met_84) ).T)
plt.show()