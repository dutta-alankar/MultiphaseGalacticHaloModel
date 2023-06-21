# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:51:35 2022

@author: alankar
"""

import sys

sys.path.append("")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../submodules/AstroPlasma")
import time
import os
import numpy as np
from scipy.integrate import simpson
from astro_plasma import EmissionSpectrum
from misc.template import modified_field
from observable.CoordinateTrans import toGalC 

sys.path.append("..")
from misc.constants import kpc, mH, mp, kB, Xp


def _fourPiNujNu_cloudy(
    nH, temperature, metallicity, redshift, indx=3000, cleanup=False, mode="PIE"
):
    background = f"\ntable HM12 redshift {redshift:.2f}" if mode == "PIE" else ""
    stream = f"""
# ----------- Auto generated from generateCloudyScript.py -----------------
#Created on {time.ctime()}
#
#@author: alankar
#
CMB redshift {redshift:.2f}{background}
sphere
radius 150 to 151 linear kiloparsec
##table ISM
abundances "solar_GASS10.abn"
metals {metallicity:.2e} linear
hden {np.log10(nH):.2f} log
constant temperature, T={temperature:.2e} K linear
stop zone 1
iterate convergence
age 1e9 years
##save continuum "spectra.lin" units keV no isotropic
save diffuse continuum "{"emission_%s.lin"%mode if indx is None else "emission_%s_%09d.lin"%(mode,indx)}" units keV
    """
    stream = stream[1:]

    if not (os.path.exists("./auto")):
        os.system("mkdir -p ./auto")
    if not (os.path.isfile("./auto/cloudy.exe")):
        os.system("cp ./cloudy.exe ./auto")
    if not (os.path.isfile("./auto/libcloudy.so")):
        os.system("cp ./libcloudy.so ./auto")

    filename = "autoGenScript_%s_%09d.in" % (mode, indx)
    with open("./auto/%s" % filename, "w") as text_file:
        text_file.write(stream)

    os.system("cd ./auto && ./cloudy.exe -r %s" % filename[:-3])

    data = np.loadtxt(
        "./auto/emission_%s.lin" % mode
        if indx is None
        else "./auto/emission_%s_%09d.lin" % (mode, indx)
    )
    if cleanup:
        os.system("rm -rf ./auto/")

    return np.vstack((data[:, 0], data[:, -1])).T


def SB_gen(redisProf: modified_field) -> np.ndarray:
    mode = redisProf.ionization
    redshift = redisProf.redshift

    _do_unmodified = False
    _use_cloudy = False
    L = 230.0
    B = 30.0
    #################################################################
    R200 = redisProf.unmodified.rCGM * redisProf.unmodified.UNIT_LENGTH / kpc 
    SuntoGC = 8.0  # kpc
    costheta = np.cos(np.deg2rad(L)) * np.cos(np.deg2rad(B))
    root1 = np.abs(
            SuntoGC
            * costheta
            * (1 + np.sqrt(1 + (R200**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
        )
    root2 = np.abs(
            SuntoGC
            * costheta
            * (1 - np.sqrt(1 + (R200**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
        )
    root_large = np.select([root1 > root2, root1 < root2], [root1, root2])
    root_small = np.select([root1 < root2, root1 > root2], [root1, root2])
    B_cond = np.logical_and(B > -90, B < 90)
    L_cond = np.logical_or(
            np.logical_and(L > 0, L < 90), np.logical_and(L > 270, L < 360)
        )
    large_root_select = np.logical_and(B_cond, L_cond)
    integrateTill = np.select(
            [large_root_select, np.logical_not(large_root_select)],
            [root_large, root_small],
            default=R200,
        )
    LOSsample = np.logspace(-3, np.log10(integrateTill+0.1), 90) 
    rad, phi, theta = toGalC(L, B, LOSsample)
    height = np.abs(rad*np.cos(np.deg2rad(theta)))
    radius = np.abs(rad*np.sin(np.deg2rad(theta)))  #along disk
    ##########################################################
 
    SB_hot = np.zeros_like(EmissionSpectrum._energy)
    SB_warm = np.zeros_like(EmissionSpectrum._energy)
    
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same

    (
        nhot_local,
        nwarm_local,
        nhot_global,
        nwarm_global,
        fvw,
        fmw,
        prs_hot,
        prs_warm,
        Tcut,
    ) = redisProf.ProfileGen(radius)
    metallicity = redisProf.unmodified.metallicity
    xmin = redisProf.xmin

    Temp = redisProf.TempDist

    ThotM = redisProf.prs_hot / (redisProf.nhot_local * kB)

    for indx, r_val in enumerate(radius):
        _, gvh, gvw = redisProf.probability_ditrib_mod(
            ThotM=ThotM[indx],
            fvw=fvw[indx],
            Temp=Temp,
            xmin=xmin[indx],
            Tcutoff=Tcut[indx],
        )
        TmedVH = ThotM[indx] * np.exp(redisProf.sigH**2 / 2)
        xh = np.log(Temp / TmedVH)
        xw = np.log(Temp / redisProf.TmedVW)

        # Assumption nT and \rho T are all constant
        nHhot = (
            redisProf.nHhot_local[indx]
            * redisProf.TmedVH[indx]
            * np.exp(-redisProf.sigH**2 / 2)
            / Temp
        )  # CGS
        nHwarm = (
            redisProf.nHwarm_local[indx]
            * redisProf.TmedVW
            * np.exp(-redisProf.sigW**2 / 2)
            / Temp
        )  # CGS

        r_val = (
            0.5
            * ((LOSsample[indx] + LOSsample[indx - 1]) if indx != 0 else LOSsample[indx])
            * kpc
        )
        dr_val = (
            (LOSsample[indx] - LOSsample[indx - 1]) if indx != 0 else LOSsample[indx]
        ) * kpc

        # 2d array row-> Temperature & column -> energy
        fourPiNujNu_hot = np.array(
            [
                fourPiNujNu(nHhot[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2
                for i in range(Temp.shape[0])
            ]
        )

        fourPiNujNu_warm = np.array(
            [
                fourPiNujNu(nHwarm[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2
                for i in range(Temp.shape[0])
            ]
        )

        hotInt = np.array(
            [simpson(fourPiNujNu_hot[:, i] * gvh, xh) for i in range(energy.shape[0])]
        ) # * (1 - redisProf.fvw[indx])  # global density sensitive

        warmInt = np.array(
            [simpson(fourPiNujNu_warm[:, i] * gvw, xw) for i in range(energy.shape[0])]
        ) # * redisProf.fvw[indx]  # global density sensitive

        # erg/s/cm^2/deg^2/keV
        SB_hot += hotInt * dr_val
        SB_warm += warmInt * dr_val
    
    return np.vstack((energy, SB_hot, SB_warm)).T
    
    

