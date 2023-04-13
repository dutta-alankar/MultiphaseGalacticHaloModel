#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:25:17 2023

@author: alankar
"""
import matplotlib
import matplotlib.pyplot as plt
import pickle

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
plt.rcParams['font.size'] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True


def plot_profile(unmod, mod, ionization):
    with open(f'figures/mod_prof_{unmod}_{mod}_{ionization}.pickle','rb') as data_file:
        label = "IC" if mod == "isochor" else "IB"
        profile = pickle.load(data_file)
        radius = profile["radius"]
        nhot_local = profile['nhot_local']
        nhot_global = profile["nhot_global"]
        nwarm_local = profile["nwarm_local"]
        nwarm_global = profile["nwarm_global"]
        
        pl1 = plt.loglog(radius, nhot_local, label=r"$<n^{(h)}>$ (%s)"%label, linestyle='--')
        pl2 = plt.loglog(radius, nwarm_local, label=r"$<n^{(w)}>$ (%s)"%label, linestyle='--')
        
        if (unmod=="isoth" and mod=="isobar"):     
            plt.loglog(radius, nhot_global, label=r"$<n^{(h)}>_g$ (%s)"%label, linestyle='-', color=pl1[0].get_color(), linewidth=2)
            plt.loglog(radius, nwarm_global, label=r"$<n^{(w)}>_g$ (%s)"%label, linestyle='-', color=pl2[0].get_color(), linewidth=2)
        else:
            plt.loglog(radius, nhot_global, label=r"$<n^{(h)}>_g$ (%s)"%label, linestyle='-', color=pl1[0].get_color())
            plt.loglog(radius, nwarm_global, label=r"$<n^{(w)}>_g$ (%s)"%label, linestyle='-', color=pl2[0].get_color())

unmod = "isoth"
mod = "isochor"
ionization = "PIE"
plt.figure(figsize=(13,10))
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isobar"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
plt.legend(loc="upper right", ncol=3, fontsize=18)  
plt.xlim(xmin=8, xmax=300)  
plt.ylim(ymin=1e-5, ymax=1e-2)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)     
# plt.show()

plt.figure(figsize=(13,10))
unmod = "isent"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isochor"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)  
plt.legend(loc="upper right", ncol=3, fontsize=18)
# plt.xlim(xmin=8, xmax=300)  
# plt.ylim(ymin=1e-5, ymax=1e-2)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)           
# plt.show()
        