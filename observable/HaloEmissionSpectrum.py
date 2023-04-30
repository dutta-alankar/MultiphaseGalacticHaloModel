# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:51:35 2022

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
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


def luminosity_spectrum_gen(redisProf: modified_field) -> np.ndarray:
    mode = redisProf.ionization
    redshift = redisProf.redshift

    _do_unmodified = False
    _use_cloudy = False

    radius = np.linspace(
        redisProf.unmodified.Halo.r0 * redisProf.unmodified.Halo.UNIT_LENGTH / kpc,
        redisProf.unmodified.rCGM * redisProf.unmodified.UNIT_LENGTH / kpc + 0.1,
        20,
    )

    luminosityTot = np.zeros_like(EmissionSpectrum._energy)

    if _do_unmodified and not (_use_cloudy):
        fourPiNujNu = EmissionSpectrum.interpolate_spectrum

        (
            unmod_rho,
            unmod_prsTh,
            _,
            _,
            unmod_prsTot,
            unmod_nH,
            unmod_mu,
        ) = redisProf.unmodified.ProfileGen(
            radius
        )  # CGS
        metallicity = redisProf.unmodified.metallicity
        unmod_T = (unmod_prsTh / kB) / (unmod_rho / (unmod_mu * mp))
        unmod_nH = unmod_rho * Xp(metallicity) / mH

        for indx, r_val in enumerate(radius):
            rad = (
                r_val * kpc
            )  # 0.5*((radius[indx]+radius[indx-1]) if indx!=0 else 0.5*radius[indx])*kpc
            dr_val = (
                (radius[-1] - radius[-2])
                if indx == len(radius) - 1
                else (radius[indx + 1] - radius[indx]) * kpc
            )

            luminosityTot += (
                (
                    fourPiNujNu(
                        unmod_nH[indx], unmod_T[indx], metallicity[indx], redshift
                    )[:, 1]
                )
                * (4 * np.pi)
                * dr_val
                * rad**2
            )

        return np.vstack(
            (EmissionSpectrum._energy, luminosityTot / EmissionSpectrum._energy)
        ).T

    if _do_unmodified and (_use_cloudy):
        energy = None
        for indx, r_val in enumerate(radius):
            (
                unmod_rho,
                unmod_prsTh,
                _,
                _,
                unmod_prsTot,
                unmod_nH,
                unmod_mu,
            ) = redisProf.unmodified.ProfileGen(
                radius
            )  # CGS
            metallicity = redisProf.unmodified.metallicity
            unmod_T = (unmod_prsTh / kB) / (unmod_rho / (unmod_mu * mp))
            unmod_nH = unmod_rho * Xp(metallicity) / mH

            rad = (
                r_val * kpc
            )  # 0.5*((radius[indx]+radius[indx-1]) if indx!=0 else 0.5*radius[indx])*kpc
            dr_val = (
                (radius[-1] - radius[-2])
                if indx == len(radius) - 1
                else (radius[indx + 1] - radius[indx]) * kpc
            )

            data = _fourPiNujNu_cloudy(
                unmod_nH[indx],
                unmod_T[indx],
                metallicity[indx],
                redshift,
                indx=indx,
                cleanup=True if indx == (radius.shape[0] - 1) else False,
                mode=mode,
            )
            energy = data[:, 0]
            luminosityTot += data[:, 1] * (4 * np.pi) * dr_val * rad**2

        return np.vstack((energy, luminosityTot / energy)).T

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

        # divide by energy to get photon count
        # 2d array row-> Temperature & column -> energy
        fourPiNujNu_hot = np.array(
            [
                fourPiNujNu(nHhot[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]
                / energy
                for i in range(Temp.shape[0])
            ]
        )

        fourPiNujNu_warm = np.array(
            [
                fourPiNujNu(nHwarm[i], Temp[i], metallicity[indx], redshift, mode)[:, 1]
                / energy
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
        luminosityTot += (hotInt + warmInt) * (4 * np.pi) * dr_val * r_val**2

    return np.vstack((energy, luminosityTot)).T
