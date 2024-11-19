# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:31:38 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib import gridspec, colors, rcParams
from itertools import product
import pickle
import os, sys
import matplotlib.patches as patches

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rcParams['hatch.linewidth'] = 2.5  # previous svg hatch linewidth
# plt.style.use('dark_background')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def plot_map(unmod: str, mod: str, ionization: str, map_type: str) -> None:
    disk_incl = True
    ratio = False
    cmap = "terrain" 
    print(unmod, mod, ionization, map_type, ((' with ' if disk_incl else ' without ') + 'disk'))

    title = ""
    if unmod=="isoth":
        title += r"$\gamma = 1$ polytrope"
    else:
        title += r"$\gamma=5/3$ polytrope"
    if mod == "isochor":
        title += r" [IC]"
    else:
        title += r" [IB]"
    if not ratio:
        title += " + Disk" if disk_incl else ""

    cblabel = ""
    if map_type == "emission":
        if not ratio:
            cblabel = r"$\rm EM\ [ cm^{-6} pc]$"
        else:
            cblabel = f"(EM {title})/(EM Disk)" 
        vmin = 0.007 if disk_incl else 0.001
        vmax = 0.10 if disk_incl else 0.02
        if ratio:
            vmin = 0.18 if unmod =="isoth" else 0.045
            vmax = 1.54 if unmod == "isoth" else 0.49
    else:
        if not ratio:
            cblabel = r"$\rm DM\ [ cm^{-3} pc]$"
        else:
            cblabel = f"(DM {title})/(DM Disk)"
        vmin = 19.5 if disk_incl else 12.8
        vmax = 66.4 if disk_incl else 39
        if ratio:
            vmin = 0.6 if unmod == "isoth" else 0.3
            vmax = 4.3 if unmod == "isent" else 2.4

    with open(
        f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.pickle", "rb"
    ) as data_file:
        data = pickle.load(data_file)
        # print(list(data.keys()))
        l = data["l"]
        b = data["b"]
        map_val = data["map"]
        disk_val = data["disk"]
        quantity = map_val + (disk_val if disk_incl else 0.)
        if ratio:
            quantity = map_val/disk_val

        # eRosita bubble
        mask = False
        cutout = np.logical_and(np.logical_or(l<=45, l>=315),
                                np.logical_and(b>=-30, b<=30))
        condition = np.logical_and(np.logical_not(cutout), np.logical_not(np.isnan(quantity)))
        print("min:", np.min(quantity[condition]) )
        print("max:", np.max(quantity[condition]) )

    # Make plot
    levels = 200
    l_mod = l - 360  # np.select([l<=180,l>180],[l,l+360])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="mollweide")

    params = {"norm": colors.LogNorm(vmin=vmin, vmax=vmax)} \
             if map_type=="emission" else {"vmin": vmin, "vmax": vmax,}
    if ratio:
        params = {"vmin": vmin, "vmax": vmax,}
    if (map_type == "emission" or "dispersion"):
      cs = ax.pcolormesh(
        np.deg2rad(l_mod),
        np.deg2rad(b),
        np.ma.masked_array(quantity, mask=np.logical_not(condition)) if \
            mask else quantity, 
        cmap=cmap,
        **params)
      cs = ax.pcolormesh(
        np.deg2rad(-l_mod),
        np.deg2rad(b),
        np.ma.masked_array(quantity, mask=np.logical_not(condition)) if \
            mask else quantity, 
        cmap=cmap,
        **params)
    else :
      cs = ax.pcolormesh(
        np.deg2rad(l_mod),
        np.deg2rad(b),
        np.ma.masked_array(quantity, mask=np.logical_not(condition)) if \
            mask else quantity, 
        cmap=cmap,
        **params)
      cs = ax.pcolormesh(
        np.deg2rad(-l_mod),
        np.deg2rad(b),
        np.ma.masked_array(quantity, mask=np.logical_not(condition)) if \
            mask else quantity, 
        cmap=cmap,
        **params)
    ax.grid(True)
    if ratio:
        cbar = fig.colorbar(
            cs,
            pad=0.05,
            orientation="horizontal",
            shrink=1.20,
            aspect=40,
            location="top",
            format="%.2f",
            extend="both",
        )
    else:
        cbar = fig.colorbar(
        cs,
        pad=0.05,
        orientation="horizontal",
        shrink=1.20,
        aspect=40,
        location="top",
        extend="both",
        )
    if map_type == "emission":
       ax.scatter([np.deg2rad(230-360),],[np.deg2rad(30.),], marker=r'$\bigotimes$', 
                  # ec="white", 
                  c="#ff3131", s=200, linewidths=0.5)
       if not(ratio):
           cbar.ax.scatter([3.0e-02,], [0.5,], marker=r'$\bigotimes$', c="#ff3131", s=170, linewidths=0.5)
           # Ponti obs point
           obs_reg = np.logical_and(l>=229, l<=231)
           obs_reg = np.logical_and(obs_reg, b>=29)
           obs_reg = np.logical_and(obs_reg, b<=31)
           print("Ponti model pred: ", np.mean(quantity[obs_reg]))
    if map_type == "dispersion":   
       ax.scatter([np.deg2rad(142),],[np.deg2rad(41.),], marker=r'$\bigotimes$', 
                  # ec="white",
                  c="#ff3131", s=200, linewidths=0.5)
       if not (ratio):
           cbar.ax.scatter([3.0e+01,], [0.5,], marker=r'$\bigotimes$', c="#ff3131", s=170, linewidths=0.5)
           # Bhardwaj obs point
           obs_reg = np.logical_and(l>=141, l<=143)
           obs_reg = np.logical_and(obs_reg, b>=40)
           obs_reg = np.logical_and(obs_reg, b<=42)
           print("Bhardwaj model pred: ", np.mean(quantity[obs_reg]))
       
       
    cbar.ax.tick_params(which='major', labelsize=24)
    cbar.ax.tick_params(which='minor', labelsize=22)
    cbar.ax.tick_params(which='major', length=10, width=4)
    cbar.ax.tick_params(which='minor', length=5, width=1.5)
    if ratio:
        cbar.set_label(cblabel, rotation=0, labelpad=8, fontsize=26, color="royalblue")
    else:
        cbar.set_label(cblabel, rotation=0, labelpad=8, fontsize=26, color="black")

    fig.tight_layout()
    # plt.grid(color="gray", linestyle=":", linewidth=1.0, alpha=1.0, zorder=256)
    plt.grid(color="white", linestyle=":", linewidth=1.0, alpha=1.0, zorder=256)
    plt.tick_params(axis="both", which="major", length=14, width=3, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=10, width=2, labelsize=22)
    ax.tick_params(axis="x", colors="black") #colors="black")

    major_axis_labels = ax.xaxis.get_majorticklabels()
    with HiddenPrints():
        plt.setp(major_axis_labels)
    # Create offset transform by 5 points in x direction
    # Matplotlib figures use 72 points per inch (ppi).
    # So to to shift something by x points, you may shift it by x/72 inch.
    dx = 0 / 72.0
    dy = -30 / 72.0
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in major_axis_labels:
        label.set_transform(label.get_transform() + offset)

    axes_labels = ax.get_xticks().tolist()
    axes_labels = list(
        np.round(
            np.rad2deg(
                np.select(
                    [np.array(axes_labels) < 0, np.array(axes_labels) > 0],
                    [np.array(axes_labels) + 2 * np.pi, np.array(axes_labels)],
                )
            )
        )
    )
    # axes_labels = [r'$%d^{{\fontsize{50pt}{3em}\selectfont{}\circ}}$'%label for label in axes_labels]
    axes_labels = [r"$%d^{\circ}$" % label for label in axes_labels]
    ax.set_xticklabels(axes_labels)
    # plot a patch
    p = patches.Rectangle((np.deg2rad(-45), np.deg2rad(-30)), 
                          np.deg2rad(90), np.deg2rad(60), 
                          linewidth=0, fill=None, hatch='\\',
                          edgecolor='firebrick', alpha=0.7)
    ax.add_patch(p)
    if disk_incl and not ratio:
        plt.gcf().text(0.28, 0.98, title, rotation=0, rotation_mode='anchor', 
                       fontsize=26, color="royalblue")
    if not disk_incl and not ratio:
        plt.gcf().text(0.38, 0.98, title, rotation=0, rotation_mode='anchor', 
                       fontsize=26, color="royalblue")
    #plt.show()
    if not ratio:
        extra = f"{'_ndisk' if not(disk_incl) else ''}"
    else:
        extra = "_ratio"
    plt.savefig(
        f"figures/map_moll_{map_type}_{unmod}_{mod}_{ionization}{extra}.png",
        transparent=False,
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor",] # "isobar"]
    ionization = ["PIE",]# "CIE"]
    map_type = ["emission", "dispersion"]

    for condition in product(unmod, mod, ionization, map_type):
        plot_map(*condition)
