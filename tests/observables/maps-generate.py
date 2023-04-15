# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:34:18 2022

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
import pickle
from typing import Union, Optional
from itertools import product
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
from observable.DispersionMeasure import DispersionMeasure as DM
from observable.EmissionMeasure import EmissionMeasure as EM
from misc.template import unmodified_field, modified_field


def gen_measure(
    unmod: str,
    mod: str,
    ionization: str,
    l: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
    map_type: str,
) -> None:
    print(map_type, unmod, mod, ionization)

    showProgress = False

    cutoff = 4.0
    TmedVW = 3.0e5
    sig = 0.3
    redshift = 0.001

    unmodified: Optional[unmodified_field] = None
    if unmod == "isoth":
        TmedVH = 1.5e6
        THotM = TmedVH * np.exp(-(sig**2) / 2)
        unmodified = IsothermalUnmodified(
            THot=THotM,
            P0Tot=4580,
            alpha=1.9,
            sigmaTurb=60,
            M200=1e12,
            MBH=2.6e6,
            Mblg=6e10,
            rd=3.0,
            r0=8.5,
            C=12,
            redshift=redshift,
            ionization=ionization,
        )
    else:
        nHrCGM = 1.1e-5
        TthrCGM = 2.4e5
        sigmaTurb = 60
        ZrCGM = 0.3
        unmodified = IsentropicUnmodified(
            nHrCGM=nHrCGM,
            TthrCGM=TthrCGM,
            sigmaTurb=sigmaTurb,
            ZrCGM=ZrCGM,
            redshift=redshift,
            ionization=ionization,
        )

    modified: Optional[modified_field] = None
    if mod == "isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff, isobaric=0)

    if map_type == "dispersion":
        map_val = DM(modified).make_map(l, b, showProgress=showProgress)
    else:
        map_val = EM(modified).make_map(l, b, showProgress=showProgress)

    with open(f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.pickle", "wb") as f:
        data = {
            "l": l,
            "b": b,
            "map": map_val,
        }
        pickle.dump(data, f)


if __name__ == "__main__":
    unmod = [
        "isent",
    ]  # "isent"]
    mod = [
        "isobar",
    ]  # "isobar"]
    ionization = [
        "CIE",
    ]  # "CIE"]

    b = np.linspace(-90, 90, 50)
    l = np.linspace(0.0, 360, 51)

    l, b = np.meshgrid(l, b)

    for condition in product(unmod, mod, ionization):
        gen_measure(*condition, l, b, "dispersion")
        gen_measure(*condition, l, b, "emission")


"""
    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))

    fig = plt.figure()#figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

    # Set x,y ticks
    plt.xticks(l_plot)#, fontsize=8)
    plt.yticks(b_plot)#, fontsize=8)
    ax.set_rlabel_position(12)
    #ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
    ax.set_theta_zero_location('S')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [PIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()

    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [PIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM_moll-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()


    # CIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60,
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12,
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)

    dispersion_CIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('dispersion_CIE_lb-isoth-ic.npy', dispersion_CIE)


    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))

    fig = plt.figure()#figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

    # Set x,y ticks
    plt.xticks(l_plot)#, fontsize=8)
    plt.yticks(b_plot)#, fontsize=8)
    ax.set_rlabel_position(12)
    #ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
    ax.set_theta_zero_location('S')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM-isth-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()

    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM_moll-isth-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()

# ____________________________________________________________
# _________________ Isentropic profile _______________________

if(do_isentropic):
    nHrCGM = 1.1e-5
    TthrCGM = 2.4e5
    sigmaTurb = 60
    ZrCGM = 0.3
    TmedVW = 3.e5
    sig = 0.3
    cutoff = 4.0

    b = np.linspace(-90, 90, 50)
    l = np.linspace(0., 360, 51)

    l, b = np.meshgrid(l, b)


    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)

    dispersion_PIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('dispersion_PIE_lb-isent-ic.npy', dispersion_PIE)


    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))

    fig = plt.figure()#figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

    # Set x,y ticks
    plt.xticks(l_plot)#, fontsize=8)
    plt.yticks(b_plot)#, fontsize=8)
    ax.set_rlabel_position(12)
    #ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
    ax.set_theta_zero_location('S')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [PIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM-isent-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()

    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), dispersion_PIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [PIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM_moll-isent-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()


    # CIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)

    dispersion_CIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('dispersion_CIE_lb-isent-ic.npy', dispersion_CIE)


    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))

    fig = plt.figure()#figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

    # Set x,y ticks
    plt.xticks(l_plot)#, fontsize=8)
    plt.yticks(b_plot)#, fontsize=8)
    ax.set_rlabel_position(12)
    #ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
    ax.set_theta_zero_location('S')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM-isent-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()

    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), dispersion_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'DM ($\rm cm^{-3} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./DM_moll-isent-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()
"""
