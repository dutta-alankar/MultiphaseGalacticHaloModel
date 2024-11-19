# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:26:51 2023

@author: alankar
Usage: python distribution-prop.py <halo_ID>
"""

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import sys
import pickle

## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = False
matplotlib.rcParams["ytick.right"] = False
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
#matplotlib.rcParams["figure.figsize"] = (13,10)
sns.set(rc={"figure.figsize": (13, 10)})
sns.set_style("darkgrid", {"grid.linestyle": ":"})

halo_id = int(sys.argv[1])
try:
    file = h5py.File(f"halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

mass = np.log10(np.array(file["/NumberDensity"])) + np.log10(np.array(file["/Volume"]))
volume = np.log10(np.array(file["/Volume"]))
#sfr = np.array(file["/SFR"])

bins = 300

fig = plt.figure(figsize=(14,6))
g = sns.JointGrid(xlim=(-6.0, -1.6), ylim=(3.9, 6.4), marginal_ticks=True)  # x = ,
# y = r"Gas Temperature $[log\ K]$")
#print(isinstance(g,matplotlib.figure.Figure),type(fig))
condition = np.logical_and(np.log10(file["/Temperature"]) >= 3.99, np.log10(file["/Temperature"]) <= 6.4)
condition = np.logical_and(condition, np.logical_and(np.log10(file["/nH"]) >= -6.0, np.log10(file["/nH"]) <= -1.5))
#condition = np.logical_and(condition, sfr<=0)

#luminosity = np.log10(-np.array(file["/nH"])[condition]**2 * np.array(file["/Lambda"])[condition] * np.array(file["/Volume"])[condition])


"""
ax = sns.scatterplot(x=np.log10(file["/nH"]),
             y=np.log10(file["/Temperature"]),
             hue=mass,
             palette="viridis",
             ax=g.ax_joint,)

norm = plt.Normalize(np.min(mass), np.max(mass))
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)
"""

"""
ax = sns.histplot(x=np.log10(file["/nH"]),
             y=np.log10(file["/Temperature"]),
             weights=np.log10(mass),
             stat="density",
             bins=(300,301),
             ax=g.ax_joint,
             common_norm=True,
             # log_scale=True,
             cmap="viridis",
             norm = LogNorm(vmin=1e-2, vmax=1e1),
             cbar=True, cbar_kws=dict(shrink=.90, label=r"Mass weighted PDF"),)
"""

"""
scatter=g.ax_joint.scatter(np.log10(file["/nH"]),
                   np.log10(file["/Temperature"]),
                   c=np.log10(mass/np.median(mass)),
                   s=1,
                   cmap="viridis",)
"""

counts, xedges, yedges, im = g.ax_joint.hist2d(
    x=np.log10(file["/nH"])[condition],
    y=np.log10(file["/Temperature"])[condition],
    weights=volume[condition],
    bins=(bins, bins),
    density=True,
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(),
)

data = None
with open('../data_param.pickle', 'rb') as f:
    data = pickle.load(f)

rho_h = -5.1
temp_h = np.log10(data['T_h'])
del_h = data['del_h']
sig_temp_h = data['sig_h']
sig_rho_h = sig_temp_h * (del_h-1.0)

rho_w = -4.7
temp_w = np.log10(data['T_w'])
del_w = data['del_w']
sig_temp_w = data['sig_w']
sig_rho_w = sig_temp_w * (del_w-1.0)

rho_c = -2.5
temp_c = np.log10(data['T_c'])
del_c = data['del_c']
sig_temp_c = data['sig_c']
sig_rho_c = sig_temp_c * (del_c-1.0)


g.ax_joint.plot((rho_h-sig_rho_h,rho_h+sig_rho_h), (temp_h-sig_temp_h,temp_h+sig_temp_h), 'r-')
g.ax_joint.plot(rho_h, temp_h,'k*')

g.ax_joint.plot((rho_w-sig_rho_w,rho_w+sig_rho_w), (temp_w-sig_temp_w,temp_w+sig_temp_w), 'r-')
g.ax_joint.plot(rho_w, temp_w,'k*')

g.ax_joint.plot((rho_c-sig_rho_c,rho_c+sig_rho_c), (temp_c-sig_temp_c,temp_c+sig_temp_c), 'r-')
g.ax_joint.plot(rho_c, temp_c,'k*')

vol_pdf, bin_edges = np.histogram(
    np.log10(file["/nH"])[condition],
    bins=bins,
    density=True,
    weights=np.array(file["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
g.ax_marg_x.semilogy(centers, vol_pdf)
#g.ax_marg_x.set_xlim(xmin =-6.1) 
#g.ax_marg_x.set_ylim(3.9, 7.6)

vol_pdf, bin_edges = np.histogram(
    np.log10(file["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=np.array(file["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
g.ax_marg_y.semilogx(vol_pdf, centers)

'''
sns.kdeplot(
    x= np.log10(file["/nH"])[condition],
    weights=volume[condition],
    linewidth=2,
    ax=g.ax_marg_x,
)

#g.ax_marg_x.set_yscale("log")
g.ax_marg_x.set_ylabel("")

sns.kdeplot(
    y= np.log10(file["/Temperature"])[condition],
    weights=volume[condition],
    linewidth=2,
    ax=g.ax_marg_y,
)
'''
#g.ax_marg_y.set_xscale("log")
#g.ax_marg_y.set_xlabel("")

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(g.ax_marg_y)
cax = divider.append_axes("right", size="20%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.yaxis.set_label_position("right")
cbar.ax.yaxis.tick_right()
cbar.ax.set_ylabel(r"Volume weighted distribution")

g.ax_joint.set_xlabel(r"Gas Hydrogen Density $n_H\ [log\ cm^{-3}]$")
g.ax_joint.set_ylabel(r"Gas Temperature $[log\ K]$")
plt.savefig("figures/distribution.png", bbox_inches="tight")
plt.show()
