# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:24:09 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from itertools import product
from parse_observation import observedColDens

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


def plot_column_density(unmod, mod, ion, alpha=0.5):
    states = ["PIE", "CIE"]

    for ionization in states:
        with open(
            f"figures/N_{ion}_{unmod}_{mod}_{ionization}.pickle", "rb"
        ) as data_file:
            data = pickle.load(data_file)
            impact = data["impact"]
            column_density = data[f"N_{ion}"]
            rCGM = data["rCGM"]
            if ionization == "PIE":
                plt.loglog(
                    np.array(impact) / rCGM,
                    column_density,
                    linestyle="-" if mod == "isochor" else ":",
                    alpha=1.0,
                    color="salmon" if unmod == "isoth" else "cadetblue",
                )
            else:
                plt.loglog(
                    np.array(impact) / rCGM,
                    column_density,
                    linestyle="-" if mod == "isochor" else ":",
                    alpha=alpha,
                    color="salmon" if unmod == "isoth" else "cadetblue",
                )


def make_legend(ax, obs_handles=None, obs_labels=None, alpha=0.5):
    line_ic = matplotlib.lines.Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=4.0,
        label="isochor",
    )
    line_ib = matplotlib.lines.Line2D(
        [0], [0], color="black", linestyle=":", linewidth=4.0, label="isobar"
    )
    legend_header = plt.legend(
        loc="lower left",
        prop={"size": 20},
        framealpha=0.3,
        shadow=False,
        fancybox=False,
        bbox_to_anchor=(0.06, 0.17) if obs_handles is not None else (0.06, 0.12),
        ncol=2,
        fontsize=18,
        handles=[line_ic, line_ib],
        title="Modification type",
        title_fontsize=20,
    )
    legend_header.get_frame().set_edgecolor(None)
    legend_header.get_frame().set_linewidth(0.0)
    ax.add_artist(legend_header)

    # legend_header._legend_box.align = "left"
    legend_header.get_frame().set_facecolor("white")
    legend_header.get_frame().set_edgecolor(None)
    legend_header.get_frame().set_linewidth(0.0)
    ax.add_artist(legend_header)

    red_patch = mpatches.Patch(color="salmon")
    blue_patch = mpatches.Patch(color="cadetblue")
    light_red_patch = mpatches.Patch(color="salmon", alpha=alpha)
    light_blue_patch = mpatches.Patch(color="cadetblue", alpha=alpha)

    if obs_handles is not None:
        legend = plt.legend(
            loc="lower left",
            prop={"size": 20},
            framealpha=0.3,
            shadow=False,
            fancybox=True,
            bbox_to_anchor=(0.03, 0.01),
            ncol=2,
            fontsize=18,
            handles=[
                red_patch,
                (red_patch, blue_patch),
                tuple(obs_handles[::-1]),
                blue_patch,
                (light_red_patch, light_blue_patch),
            ],
            labels=[
                "isothermal",
                "PIE",
                obs_labels[0],
                "isentropic",
                "CIE",
            ],
            handler_map={
                tuple: HandlerTuple(ndivide=None),
            },
            handlelength=1.5,
        )
    else:
        legend = plt.legend(
            loc="lower left",
            prop={"size": 20},
            framealpha=0.3,
            shadow=False,
            fancybox=True,
            bbox_to_anchor=(0.03, 0.01),
            ncol=2,
            fontsize=18,
            handles=[
                red_patch,
                (red_patch, blue_patch),
                blue_patch,
                (light_red_patch, light_blue_patch),
            ],
            labels=[
                "isothermal",
                "PIE",
                "isentropic",
                "CIE",
            ],
            handler_map={
                tuple: HandlerTuple(ndivide=None),
            },
            handlelength=1.5,
        )
    # legend._legend_box.align = "left"
    # print(legend.get_texts()[2].__dict__)
    # print(dir(legend.get_texts()[2]) )
    # help(legend.get_texts()[2].set_bbox)

    # legend.get_texts()[2].set_ha("center")
    legend.get_texts()[2].update(
        {
            "ha": "right",
            "position": (1.0, 0.5),
        }
    )
    legend.get_frame().set_edgecolor("rebeccapurple")
    legend.get_frame().set_facecolor("ivory")
    legend.get_frame().set_linewidth(1.0)
    ax.add_artist(legend)
    # print(ax.get_legend_handles_labels())
    # ax.get_legend_handles_labels()[0][-1].get_texts().set_ha("center")


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    element = "N V"  # "OVI"
    alpha = 0.3

    plt.figure(figsize=(13, 10))

    for condition in product(unmod, mod):
        plot_column_density(*condition, "".join(element.split()), alpha=alpha)

    observation = observedColDens()

    (
        gal_id_min,
        gal_id_max,
        gal_id_detect,
        rvir_select_min,
        rvir_select_max,
        rvir_select_detect,
        impact_select_min,
        impact_select_max,
        impact_select_detect,
        coldens_min,
        coldens_max,
        coldens_detect,
        e_coldens_detect,
    ) = observation.col_density_gen(element=element)
    if len(e_coldens_detect != 0):
        yerr = np.log(10) * e_coldens_detect * 10.0**coldens_detect
        plt.errorbar(
            impact_select_detect / rvir_select_detect,
            10.0**coldens_detect,
            yerr=yerr,
            fmt="o",
            color="black",
            label=r"Observations",
            markersize=10,
        )
        plt.plot(
            impact_select_min / rvir_select_min,
            10.0**coldens_min,
            "^",
            color="black",
            label=r"Observations",
            markersize=10,
        )
        plt.plot(
            impact_select_max / rvir_select_max,
            10.0**coldens_max,
            "v",
            color="black",
            label=r"Observations",
            markersize=10,
        )
        obs_handles, obs_labels = plt.gca().get_legend_handles_labels()
        make_legend(plt.gca(), obs_handles[-3:], obs_labels[-3:], alpha=alpha)
    else:
        make_legend(plt.gca(), alpha=alpha)

    if element == "O VII":
        # obs
        NOVII_obs = 15.68
        NOVII_err = 0.27
        yerr = np.log(10) * NOVII_err * 10.0**NOVII_obs
        plt.axhspan(
            2 * 10.0**NOVII_obs,
            2 * (10.0**NOVII_obs - yerr),
            color="gray",
            alpha=0.2,
            zorder=0,
        )
        plt.ylim(ymin=2e13)

    if element == "O VIII":
        # obs
        NOVII_obs = 15.68
        NOVII_err = 0.27
        NOVIII_obs = NOVII_obs - np.log10(4)
        NOVIII_err = NOVII_err - np.log10(4)
        yerr = np.log(10) * NOVIII_err * 10.0**NOVIII_obs
        plt.axhspan(
            2 * 10.0**NOVIII_obs,
            2 * (10.0**NOVIII_obs - yerr),
            color="gray",
            alpha=0.2,
            zorder=0,
        )
        plt.ylim(ymin=2e13)

    # plt.legend()
    plt.xlabel(r"Impact parameter [$r_{vir}$]", size=28)
    plt.ylabel(r"Column density of %s ($cm^{-2}$)" % element, size=28)
    plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
    plt.xlim(xmin=0.05, xmax=1.1)
    # plt.ylim(ymin=10**11.7, ymax=10.0**15.3)
    plt.savefig(f"figures/column_density_{element}.png", transparent=False)
