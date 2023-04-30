# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:30:44 2023

@author: alankar
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from itertools import product

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


def plot_profile(radius, unmod, all_mods, ionization, figure):
    fig = figure
    ax = fig.gca()
    for mod in all_mods:
        with open(
            f"figures/{unmod}_{mod}_{ionization}_distrib-r={radius:.1f}kpc.pickle", "rb"
        ) as data_file:
            profile = pickle.load(data_file)
            Tcutoff = profile["T_cutoff"]
            cutoff = profile["cutoff"]
            TmedVu = profile["T_hot_M"] * np.exp(profile["sig_u"] ** 2 / 2.0)
            TmedVW = profile["T_med_VW"]
            TempDist = profile["TempDist"]
            gv_unmod = profile["gv_unmod"]
            gvh = profile["gv_h"]
            gvw = profile["gv_w"]
            if mod == "isochor":
                ax.vlines(
                    np.log10(Tcutoff),
                    1e-3,
                    2.1,
                    colors="black",
                    linestyles="--",
                    label=r"$T_c\ (t_{\rm cool}/t_{\rm ff}=%d)$" % cutoff,
                    linewidth=3,
                    zorder=20,
                    alpha=0.6,
                )
                ax.vlines(
                    np.log10(TmedVu),
                    1e-3,
                    2.1,
                    colors="tab:red",
                    linestyles=":",
                    label=r"$T_{med,V}^{(h)}$",
                    linewidth=3,
                    zorder=30,
                )
                ax.vlines(
                    np.log10(TmedVW),
                    1e-3,
                    2.1,
                    colors="tab:blue",
                    linestyles=":",
                    label=r"$T_{med,V}^{(w)}$",
                    linewidth=3,
                    zorder=40,
                )
                ax.semilogy(
                    np.log10(TempDist),
                    gv_unmod,
                    color="tab:red",
                    alpha=0.5,
                    label="hot, unmod",
                    linewidth=5,
                    zorder=6,
                )
            ax.semilogy(
                np.log10(TempDist),
                gvh,
                color="tab:orange" if mod == "isobar" else "tab:red",
                label=f"hot, mod, {'IB' if mod == 'isobar' else 'IC'}",
                linewidth=5,
                zorder=5,
            )
            ax.semilogy(
                np.log10(TempDist),
                gvw,
                color="tab:blue" if mod == "isobar" else "tab:cyan",
                label=f"warm, mod, {'IB' if mod == 'isobar' else 'IC'}",
                linestyle="--",
                linewidth=5,
                zorder=7,
            )


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    all_mods = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]

    for condition in product(unmod, ionization):
        radius = 20.0  # 200.0 if unmod == "isoth" else 20.0  # kpc
        fig = plt.figure(figsize=(13, 10))
        plot_profile(
            radius=radius,
            unmod=condition[0],
            all_mods=all_mods,
            ionization=condition[1],
            figure=fig,
        )
        plt.title(
            r"Modified probability distribution at $r = %.1f$ kpc (%s)"
            % (radius, condition[1]),
            size=28,
        )
        plt.ylim(1e-3, 2.4)
        plt.xlim(4.9, 7)
        plt.ylabel(r"$T \mathscr{P}_V(T)$", size=28)
        plt.xlabel(r"$\log_{10} (T [K])$", size=28)
        plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
        plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)

        current_handles, current_labels = plt.gca().get_legend_handles_labels()

        legend1 = plt.legend(
            loc="upper left",
            prop={"size": 20},
            framealpha=0.6,
            shadow=False,
            fancybox=False,
            bbox_to_anchor=(-0.01, 0.97),
            ncol=1,
            fontsize=18,
            handles=current_handles[:3],
            labels=current_labels[:3],
            handlelength=1.5,
        )
        # legend1.get_frame().set_edgecolor(None)
        # legend1.get_frame().set_facecolor(None)
        legend1.get_frame().set_linewidth(0.0)
        plt.gca().add_artist(legend1)

        legend2 = plt.legend(
            loc="upper right",
            prop={"size": 20},
            framealpha=0.8,
            shadow=False,
            fancybox=True,
            bbox_to_anchor=(1.06, 0.98),
            ncol=1,
            fontsize=18,
            handles=[
                current_handles[3],
                current_handles[4],
                current_handles[6],
                current_handles[5],
                current_handles[7],
            ],
            labels=[
                current_labels[3],
                current_labels[4],
                current_labels[6],
                current_labels[5],
                current_labels[7],
            ],
        )
        legend2.get_frame().set_edgecolor("rebeccapurple")
        legend2.get_frame().set_facecolor("ivory")
        legend2.get_frame().set_linewidth(1.0)
        plt.savefig(
            f"figures/PDF_{condition[0]}_{condition[1]}.png",
            transparent=False,
            bbox_extra_artists=(legend1, legend2),
            bbox_inches="tight",
        )
        # Note that the bbox_extra_artists must be an iterable
        # plt.show()
