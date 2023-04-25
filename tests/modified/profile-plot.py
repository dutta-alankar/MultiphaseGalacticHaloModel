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
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True


def plot_profile(unmod, mod, ionization):
    with open(f"figures/mod_prof_{unmod}_{mod}_{ionization}.pickle", "rb") as data_file:
        label = "IC" if mod == "isochor" else "IB"
        profile = pickle.load(data_file)
        radius = profile["radius"]
        r200 = 212  # profile["rvir"]
        nhot_local = profile["nhot_local"]
        nhot_global = profile["nhot_global"]
        nwarm_local = profile["nwarm_local"]
        nwarm_global = profile["nwarm_global"]

        pl1 = plt.loglog(
            radius / r200, nhot_local, label=r"$<n^{(h)}>$ (%s)" % label, linestyle="--"
        )
        pl2 = plt.loglog(
            radius / r200,
            nwarm_local,
            label=r"$<n^{(w)}>$ (%s)" % label,
            linestyle="--",
        )

        if unmod == "isoth" and mod == "isobar":
            plt.loglog(
                radius / r200,
                nhot_global,
                label=r"$<n^{(h)}>_g$ (%s)" % label,
                linestyle="-",
                color=pl1[0].get_color(),
                # linewidth=2,
            )
            plt.loglog(
                radius / r200,
                nwarm_global,
                label=r"$<n^{(w)}>_g$ (%s)" % label,
                linestyle="-",
                color=pl2[0].get_color(),
                # linewidth=2,
            )
        else:
            plt.loglog(
                radius / r200,
                nhot_global,
                label=r"$<n^{(h)}>_g$ (%s)" % label,
                linestyle="-",
                color=pl1[0].get_color(),
            )
            plt.loglog(
                radius / r200,
                nwarm_global,
                label=r"$<n^{(w)}>_g$ (%s)" % label,
                linestyle="-",
                color=pl2[0].get_color(),
            )


plt.figure(figsize=(13, 10))
unmod = "isoth"
mod = "isochor"
ionization = "PIE"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isobar"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
plt.legend(
    loc="upper left",
    prop={"size": 20},
    framealpha=0.3,
    shadow=False,
    fancybox=True,
    bbox_to_anchor=(-0.1, 1.15),
    ncol=4,
    fontsize=18,
)
# plt.xlim(xmin=8, xmax=300)
# plt.ylim(ymin=1e-5, ymax=1e-2)
plt.xlabel(r"$r/r_{vir}$", size=28)
plt.ylabel(r"Particle number density $(cm^{-3})$", size=28)
plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)
# plt.show()

plt.figure(figsize=(13, 10))
unmod = "isent"
mod = "isochor"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isobar"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
plt.legend(
    loc="upper left",
    prop={"size": 20},
    framealpha=0.3,
    shadow=False,
    fancybox=True,
    bbox_to_anchor=(-0.1, 1.15),
    ncol=4,
    fontsize=18,
)
# plt.xlim(xmin=8, xmax=300)
# plt.ylim(ymin=1e-5, ymax=1e-2)
plt.xlabel(r"$r/r_{vir}$", size=28)
plt.ylabel(r"Particle number density $(cm^{-3})$", size=28)
plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)
# plt.show()
