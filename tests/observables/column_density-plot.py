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
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerTuple
from itertools import product
from parse_observation import observedColDens
import sys

sys.path.append("../..")
from obs_data import parse_cgm2, parse_magiicat, parse_cubs

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
PvsC = False


def plot_column_density(unmod, mod, ion, alpha=0.5):
    states = (
        ["PIE", "CIE"]
        if PvsC
        else [
            "PIE",
        ]
    )

    for ionization in states:
        with open(
            f"figures/N_{ion}_{unmod}_{mod}_{ionization}.pickle", "rb"
        ) as data_file:
            data = pickle.load(data_file)
            impact = data["impact"]
            column_density = data[f"N_{ion}"]
            
            print(unmod)
            print(mod)
            print(ionization)
            print(f"{ion} max :", "{:e}".format(max(column_density)))
            print(f"{ion} min :", "{:e}".format(column_density[-2]))
            
            rCGM = data["rCGM"]
            unmod_file = f"../unmodified/cdens_profile-{''.join(element.split())}-unmod_{unmod}_{ionization}.txt"
            if ionization == "PIE":
                plt.loglog(
                    np.array(impact) / (rCGM/1.1),
                    column_density,
                    linestyle="-" if mod == "isochor" else ":",
                    alpha=1.0,
                    color="salmon" if unmod == "isoth" else "cadetblue",
                )
                unmod_model = np.loadtxt(unmod_file)
                plt.loglog(
                    unmod_model[:,0],
                    unmod_model[:,1],
                    linestyle=(8, (10, 3)),
                    alpha=0.5,
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
                unmod_model = np.loadtxt(unmod_file)
                plt.loglog(
                    unmod_model[:,0],
                    unmod_model[:,1],
                    linestyle=(8, (10, 3)),
                    alpha=0.5,
                    color="salmon" if unmod == "isoth" else "cadetblue",
                )


def make_legend(ax, obs_handles=None, obs_labels=None, alpha=0.5):
    line_ic = matplotlib.lines.Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=4.0,
        label="isochoric redistribution",
    )
    line_ib = matplotlib.lines.Line2D(
        [0],
        [0],
        color="black",
        linestyle=":",
        linewidth=4.0,
        label="isobaric redistribution",
    )
    line_unmod = matplotlib.lines.Line2D(
        [0],
        [0],
        color="black",
        linestyle=(8, (10, 3)),
        linewidth=4.0,
        label="unmodified profile",
    )
    if PvsC:
        type_pos = (0.06, 0.19) if obs_handles is not None else (0.06, 0.12)
    else:
        type_pos = (0.06, 0.12)

    legend_header = plt.legend(
        loc="lower left",
        prop={"size": 20},
        framealpha=0.3,
        shadow=False,
        fancybox=False,
        bbox_to_anchor=type_pos,
        ncol=1,
        fontsize=18,
        handles=[line_unmod, line_ic, line_ib],
        # title="Modification",
        # title_fontsize=20,
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
            ]
            if PvsC
            else [
                red_patch,
                tuple(obs_handles[::-1]),
                blue_patch,
            ],
            labels=[
                r"$\gamma = 1$",
                "PIE",
                obs_labels[0],
                r"$\gamma = 5/3$",
                "CIE",
            ]
            if PvsC
            else [
                r"$\gamma = 1$",
                obs_labels[0],
                r"$\gamma = 5/3$",
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
            bbox_to_anchor=(0.08, 0.01) if PvsC else (0.08, 0.05),
            ncol=2,
            fontsize=18,
            handles=[
                red_patch,
                (red_patch, blue_patch),
                blue_patch,
                (light_red_patch, light_blue_patch),
            ]
            if PvsC
            else [
                red_patch,
                blue_patch,
            ],
            labels=[
                r"$\gamma = 1$",
                "PIE",
                r"$\gamma = 5/3$",
                "CIE",
            ]
            if PvsC
            else [
                r"$\gamma = 1$",
                r"$\gamma = 5/3$",
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
    if PvsC:
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
    element = "O VI"  # "OVI"
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
    
    # print("Debug: ", 10**coldens_min)
    # print("Debug: ", 10**coldens_max)
    # print("{:e}".format(max(10**coldens_detect)))
    # print("{:e}".format(min(10**coldens_detect)))
    
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
        # CGM^2 obs
        impact_cgm2, col_dens_cgm2, col_dens_err_cgm2 = parse_cgm2.parse_col_dens_cgm2()
        condition = np.logical_and(col_dens_err_cgm2==-1, impact_cgm2<1.2)
        plt.plot(
            impact_cgm2[condition],
            10.0**(col_dens_cgm2[condition]),
            "v",
            color="black",
            markersize=10,
            markerfacecolor="None",
        )
        condition = np.logical_and(col_dens_err_cgm2!=-1, impact_cgm2<1.2)
        plt.plot(
            impact_cgm2[condition],
            10.0**(col_dens_cgm2[condition]),
            "o",
            color="black",
            markersize=10,
            markerfacecolor="None",
        )
        impact_cubs, col_dens_cubs, col_dens_err_cubs = parse_cubs.parse_col_dens_cubs("OVI")
        condition1 = np.logical_and(col_dens_err_cubs[0,:]!=-1, col_dens_err_cubs[0,:]!=1)
        condition2 = np.logical_and(col_dens_err_cubs[1,:]!=-1, col_dens_err_cubs[1,:]!=1)
        condition3 = col_dens_cubs > 0.
        condition4 = np.logical_and(col_dens_err_cubs[0,:]<=0.4, col_dens_err_cubs[1,:]<=0.4)
        condition5 = impact_cubs < 1.2
        condition = np.logical_and(condition1, condition2)
        condition = np.logical_and(condition,  condition3)
        condition = np.logical_and(condition,  condition4)
        condition = np.logical_and(condition,  condition5)
        yerr = np.log(10) * np.vstack( (col_dens_err_cubs[0,:][condition], col_dens_err_cubs[1,:][condition]) ) \
                              * 10.0**np.vstack( (col_dens_cubs[condition], col_dens_cubs[condition]) )
        ebar = plt.errorbar(
            impact_cubs[condition],
            10.0**col_dens_cubs[condition],
            yerr=yerr,
            fmt="o",
            color="gray",
            # label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
            markersize=10,
            # markerfacecolor="None",
            markeredgecolor = "black",
            zorder=200,
             # hatch="/",
        )
        ebar[2][0].set_path_effects([path_effects.withStroke(linewidth=6, foreground='black', capstyle="round"),
                                        path_effects.Normal()])
        plt.plot(
            impact_cubs[np.logical_not(condition)],
            10.0**(col_dens_cubs[np.logical_not(condition)]),
            "v",
            color="gray",
            markersize=10,
            markeredgecolor="black",
        )
    else:
        if "".join(element.split()) == "OVII":
            plt.errorbar(
                np.array([ 116.8 ])/ 200.4, # Mathur+23 Sec 2.1 & 3.1
                np.array([ 4.9e+15 ]),
                yerr=np.array([ 1.6e+15 ]),
                fmt="o",
                color="black",
                label=r"Observations",
                markersize=10,
            )
            obs_handles, obs_labels = plt.gca().get_legend_handles_labels()
            make_legend(plt.gca(), obs_handles[-3:], obs_labels[-3:], alpha=alpha)
        else:
            make_legend(plt.gca(), alpha=alpha)
    if "".join(element.split()) == "OVI":
        plt.ylim(ymin=1.2e+12, ymax=2.9e+15)
    if "".join(element.split()) == "NV":
        plt.ylim(ymin=8.5e+10, ymax=5.5e+14)

    if "".join(element.split()) == "OVII":
        # obs Gupta+12 Tab. 2 & end of Sec. 3
        NOVII_obs_min = 15.82
        NOVII_obs_max = 16.05
        plt.axhspan(
            2 * (10.0**NOVII_obs_min),
            2 * (10.0**NOVII_obs_max),
            color="gray",
            alpha=0.2,
            zorder=0,
        )
        plt.text(0.4, 1.6e+16, "MW estimate", size=22)
        plt.ylim(ymin=8.5e13, ymax=4.5e+16)
        # print("{:e}".format(2 * 10.0**NOVII_obs),"{:e}".format( 2 * (10.0**NOVII_obs - yerr)))
        
    if "".join(element.split()) == "OVIII":
        # obs
        NOVII_obs_min = 16.22
        NOVII_obs_max = 16.23
        EW_OVII_Ka_min  = 9.4
        EW_OVIII_Ka_min = 1.8 # from same OVII LOS
        EW_OVII_Ka_max  = 48.3
        EW_OVIII_Ka_max = 28.8 # from same OVII LOS
        EW_OVII_Kb_min = 3.8
        EW_OVIII_Ka_min = 9.5 # from same OVII LOS
        EW_OVII_Kb_max = 34.2
        EW_OVIII_Ka_max = 28.8 # from same OVII LOS
        fOVII_Ka  = 0.696
        fOVII_Kb  = 0.146
        fOVIII_Ka = 0.416
        lamOVII_Ka  = 21.602
        lamOVII_Kb  = 18.654
        lamOVIII_Ka = 18.967
        Kalpha = False
        if Kalpha:
            atomic = np.log10(fOVII_Ka/fOVIII_Ka) + 2*np.log10(lamOVII_Ka/lamOVIII_Ka)
        else:
            atomic = np.log10(fOVII_Kb/fOVIII_Ka) + 2*np.log10(lamOVII_Kb/lamOVIII_Ka)
        if Kalpha:
            NOVIII_obs_min = NOVII_obs_min +  + np.log10(EW_OVIII_Ka_min/EW_OVII_Ka_min)
            NOVIII_obs_max = NOVII_obs_min + atomic + np.log10(EW_OVIII_Ka_max/EW_OVII_Ka_max)
        else:
            NOVIII_obs_min = NOVII_obs_min + atomic + np.log10(EW_OVIII_Ka_min/EW_OVII_Kb_min)
            NOVIII_obs_max = NOVII_obs_min + atomic + np.log10(EW_OVIII_Ka_max/EW_OVII_Kb_max)
        # print("%e %e"%(10.**NOVIII_obs_min, 10.**NOVIII_obs_max))
        # Das+19: Discovery of a Very Hot Phase of the Milky Way Circumgalactic Medium also Tab. 2 FSM 20
        NOVIII_obs_min = np.min([NOVIII_obs_min, NOVIII_obs_max, np.log10(3.4e+15)])
        NOVIII_obs_max = np.max([NOVIII_obs_min, NOVIII_obs_max, np.log10(3.4e+15)])
        # print("%e %e"%(10.**NOVIII_obs_min, 10.**NOVIII_obs_max))
        plt.axhspan(
            2 * (10.**NOVIII_obs_min),
            2 * (10.**NOVIII_obs_max),
            color="gray",
            alpha=0.2,
            zorder=0,
        )
        plt.text(0.4, 7.6e+15, "MW estimate", size=22)
        plt.ylim(ymin=8.5e13, ymax=1.5e+16)
        # print("{:e}".format(2 * 10.0**NOVIII_obs), "{:e}".format(2 * (10.0**NOVIII_obs - yerr)))

    # plt.legend()
    plt.xlabel(r"Impact parameter b [$r_{\rm vir}$]", size=28)
    plt.ylabel(r"Column density of %s [$\rm cm^{-2}$]" % ("".join(element.split())), size=28)
    plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
    plt.xlim(xmin=0.05, xmax=1.3)
    # plt.ylim(ymin=10**11.7, ymax=10.0**15.3)
    plt.savefig(f'figures/column_density_{("".join(element.split()))}.png', transparent=False)
    plt.close()
