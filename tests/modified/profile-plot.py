# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:25:17 2023

@author: alankar
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        profile = pickle.load(data_file)
        radius = profile["radius"]
        r200 = 212  # profile["rvir"]
        nhot_local = profile["nhot_local"]
        nhot_global = profile["nhot_global"]
        nwarm_local = profile["nwarm_local"]
        nwarm_global = profile["nwarm_global"]

        plt.loglog(
            radius / r200,
            nhot_global,
            linestyle="-" if mod == "isochor" else ":",
            color="tab:red",
        )
        plt.loglog(
            radius / r200,
            nwarm_local,
            linestyle="-" if mod == "isochor" else ":",
            color="tab:cyan",
        )

        plt.loglog(
            radius / r200,
            nhot_local,
            linestyle="-" if mod == "isochor" else ":",
            color="tab:orange",
            # linewidth=2,
        )
        plt.loglog(
            radius / r200,
            nwarm_global,
            linestyle="-" if mod == "isochor" else ":",
            color="tab:blue",
            # linewidth=2,
        )


def make_legend(ax):
    line_ic = matplotlib.lines.Line2D(
        [0], [0], color="black", linestyle="-", linewidth=4.0, label="isochor"
    )
    line_ib = matplotlib.lines.Line2D(
        [0], [0], color="black", linestyle=":", linewidth=4.0, label="isobar"
    )

    legend = plt.legend(
        loc="upper right",
        prop={"size": 20},
        framealpha=0.3,
        shadow=False,
        fancybox=False,
        bbox_to_anchor=(0.96, 1.01),
        ncol=2,
        fontsize=18,
        handles=[line_ic, line_ib],
        title="Modification type",
        title_fontsize=20,
    )
    legend.get_frame().set_edgecolor(None)
    legend.get_frame().set_linewidth(0.0)
    ax.add_artist(legend)

    red_patch = mpatches.Patch(color="tab:red", label=r"$<n^{(h)}>_g$")
    blue_patch = mpatches.Patch(color="tab:blue", label=r"$<n^{(w)}>_g$")
    orange_patch = mpatches.Patch(color="tab:orange", label=r"$<n^{(h)}>$")
    cyan_patch = mpatches.Patch(color="tab:cyan", label=r"$<n^{(w)}>$")

    legend = plt.legend(
        loc="upper right",
        prop={"size": 20},
        framealpha=0.5,
        shadow=False,
        fancybox=True,
        bbox_to_anchor=(1.0, 0.9),
        ncol=2,
        fontsize=18,
        handles=[red_patch, orange_patch, blue_patch, cyan_patch],
    )
    legend.get_frame().set_edgecolor("rebeccapurple")
    legend.get_frame().set_facecolor("ivory")
    legend.get_frame().set_linewidth(1.0)


plt.figure(figsize=(13, 10))
ax = plt.gca()
unmod = "isoth"
mod = "isochor"
ionization = "PIE"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isobar"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)

# plt.xlim(xmin=8, xmax=300)
# plt.ylim(ymin=1e-5, ymax=1e-2)

make_legend(ax)

plt.xlabel(r"$r/r_{vir}$", size=28)
plt.ylabel(r"Particle number density $(cm^{-3})$", size=28)
plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)
# plt.show()

plt.figure(figsize=(13, 10))
ax = plt.gca()
unmod = "isent"
mod = "isochor"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)
mod = "isobar"
plot_profile(unmod=unmod, mod=mod, ionization=ionization)

# plt.xlim(xmin=8, xmax=300)
# plt.ylim(ymin=1e-5, ymax=1e-2)

make_legend(ax)

plt.xlabel(r"$\rm{r/r_{vir}}$", size=28)
plt.ylabel(r"Particle number density $(cm^{-3})$", size=28)
plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
plt.savefig(f"figures/{unmod}_{ionization}.png", transparent=False)
# plt.show()
