# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:51:35 2022

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../submodules/AstroPlasma")
import time
import os
import numpy as np
from scipy.integrate import simpson
from astro_plasma import EmissionSpectrum
from misc.template import modified_field

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

    radius = np.linspace(
        redisProf.unmodified.Halo.r0 * redisProf.unmodified.Halo.UNIT_LENGTH / kpc,
        redisProf.unmodified.rCGM * redisProf.unmodified.UNIT_LENGTH / kpc + 0.1,
        20,
    )

    SBTot = np.zeros_like(EmissionSpectrum._energy)
   
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
            * ((radius[indx] + radius[indx - 1]) if indx != 0 else radius[indx])
            * kpc
        )
        dr_val = (
            (radius[indx] - radius[indx - 1]) if indx != 0 else radius[indx]
        ) * kpc

        # 2d array row-> Temperature & column -> energy
        fourPiNujNu_hot = np.array(
            [
                fourPiNujNu(nHhot[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]/(4*np.pi)
                for i in range(Temp.shape[0])
            ]
        )

        fourPiNujNu_warm = np.array(
            [
                fourPiNujNu(nHwarm[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]/(4*np.pi)
                for i in range(Temp.shape[0])
            ]
        )

        hotInt = (1 - redisProf.fvw[indx]) * np.array(
            [simpson(fourPiNujNu_hot[:, i] * gvh, xh) for i in range(energy.shape[0])]
        )  # global density sensitive

        warmInt = redisProf.fvw[indx] * np.array(
            [simpson(fourPiNujNu_warm[:, i] * gvw, xw) for i in range(energy.shape[0])]
        )  # global density sensitive

        # erg/s/keV all solid angles covered
        SBTot += (hotInt + warmInt) * dr_val
    ener_l = 0
    ener_u = 0
    lim_l = 0
    lim_u = 0
    for j in range(energy.shape[0]):
        if (energy[j]<=0.3):
           ener_l = energy[j]
           lim_l = j
        elif (0.3 < energy[j]<=2.):
           ener_u = energy[j]
           lim_u = j
        else :
           break  
    Tot_SB = simpson(SBTot[lim_l:lim_u,],energy[lim_l:lim_u,]) 
     
    return Tot_SB   
