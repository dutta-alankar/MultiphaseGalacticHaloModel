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
import pickle
import numpy as np
from typing import Optional
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from astro_plasma import EmissionSpectrum
from misc.template import modified_field
from observable.CoordinateTrans import toGalC 
from mpi4py import MPI

sys.path.append("..")
from misc.constants import kpc, mH, mp, kB, Xp

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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


def SB_gen(mod:str, unmod:str, ionization:str, L: float, B: float, data_dir:str = "") -> np.ndarray:
    t_start = MPI.Wtime()
    profile: Optional[modified_field] = None
    if rank == 0:
        print("Loading modified profile ...", end=" ", flush=True)
        with open(f"{data_dir}/mod_{mod}_unmod_{unmod}_ionization_{ionization}.pickle", "rb") as data_file:
            profile = pickle.load(data_file)
    profile = comm.bcast(profile, root=0)
    comm.Barrier()
    t_stop = MPI.Wtime()
    if rank == 0:
        print("Done! Took", (t_stop-t_start), "s", flush=True)

    mode = profile.ionization
    redshift = profile.redshift

    _do_unmodified = False
    _use_cloudy = False
    #################################################################
    R200 = profile.unmodified.rCGM * profile.unmodified.UNIT_LENGTH / kpc # integration is done till rCGM not r200
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
    # LOSsample = np.logspace(-3, np.log10(integrateTill+0.1), 100)
    LOSsample = np.linspace(0., integrateTill+0.1, 100)
    '''
    rad, phi, theta = toGalC(L, B, LOSsample)
    height = np.abs(rad*np.cos(np.deg2rad(theta)))
    radius = np.abs(rad*np.sin(np.deg2rad(theta)))  # along disk
    '''
    ##########################################################
 
    SB_hot = np.zeros_like(EmissionSpectrum._energy)
    SB_warm = np.zeros_like(EmissionSpectrum._energy)
    
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same

    nhot_local = profile.nhot_local
    nwarm_local = profile.nwarm_local
    nhot_global = profile.nhot_global
    nwarm_global = profile.nwarm_global
    fvw = profile.fvw
    fmw = profile.fmw
    prs_hot = profile.prs_hot
    prs_warm = profile.prs_warm
    Tcut = profile.Tcut

    metallicity = profile.unmodified.metallicity
    xmin = profile.xmin

    Temp = profile.TempDist

    ThotM = profile.prs_hot / (profile.nhot_local * kB)

    # if rank == 0:
    #     print("Temperature:", Temp, flush=True)
    for indx in range(LOSsample.shape[0]):
        t_stop = MPI.Wtime()
        los_val = (
            0.5
            * ((LOSsample[indx] + LOSsample[indx - 1]) if indx != 0 else LOSsample[indx])
        ) # kpc
        r_val, phi_val, theta_val = toGalC(L, B, los_val)
        ds_val = (
            (LOSsample[indx] - LOSsample[indx - 1]) if indx != 0 else LOSsample[indx]
        ) * kpc
        # if rank == 0:
        #     print("LOSsample:", LOSsample[indx], flush=True)
        _, gvh, gvw = profile.probability_ditrib_mod(r_val,
            ThotM=interp1d(profile.radius, ThotM, fill_value="extrapolate"),
            fvw=interp1d(profile.radius, fvw, fill_value="extrapolate"),
            Temp=Temp,
            xmin=interp1d(profile.radius, xmin, fill_value="extrapolate"),
            Tcutoff=interp1d(profile.radius, Tcut, fill_value="extrapolate"),
        )
        TmedVH = interp1d(profile.radius, ThotM, fill_value="extrapolate")(r_val) * np.exp(profile.sigH**2 / 2)
        xh = np.log(Temp / TmedVH)
        xw = np.log(Temp / profile.TmedVW)

        # Assumption nT and \rho T are all constant
        nHhot = (
            interp1d(profile.radius, profile.nHhot_local, fill_value="extrapolate")(r_val)
            * interp1d(profile.radius, profile.TmedVH, fill_value="extrapolate")(r_val)
            * np.exp(-profile.sigH**2 / 2)
            / Temp
        )  # CGS
        nHwarm = (
            interp1d(profile.radius, profile.nHwarm_local, fill_value="extrapolate")(r_val)
            * profile.TmedVW
            * np.exp(-profile.sigW**2 / 2)
            / Temp
        )  # CGS

        # 2d array row-> Temperature & column -> energy
        # if rank == 0:
        #     print("Calculating emission from the hot phase ...", end=" ", flush=True)
        # t_start = MPI.Wtime()
        fourPiNujNu_hot = np.zeros((Temp.shape[0], energy.shape[0]))
        for i in range(rank, Temp.shape[0], size):
            if nHhot[i]>1.0e-06:
                fourPiNujNu_hot[i,:] = fourPiNujNu(nHhot[i], Temp[i], 
                                                   interp1d(profile.radius, metallicity, fill_value="extrapolate")(r_val), 
                                                   redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy # per sq deg
        tmp = np.zeros_like(fourPiNujNu_hot)
        # use MPI to get the totals
        comm.Reduce([fourPiNujNu_hot, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        tmp = comm.bcast(tmp, root=0)
        fourPiNujNu_hot = np.copy(tmp)
        # t_stop = MPI.Wtime()
        # if rank == 0:
        #     print("Done! Took", (t_stop-t_start), "s", flush=True)

        # if rank == 0:
        #     print("Calculating emission from the warm phase ...", end=" ", flush=True)
        t_start = MPI.Wtime()
        fourPiNujNu_warm = np.zeros((Temp.shape[0], energy.shape[0]))
        for i in range(rank, Temp.shape[0], size):
            if nHwarm[i]>1.0e-06:
                fourPiNujNu_warm[i,:] = fourPiNujNu(nHwarm[i], Temp[i], 
                                                    interp1d(profile.radius, metallicity, fill_value="extrapolate")(r_val), 
                                                    redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy # per sq deg
        tmp = np.zeros_like(fourPiNujNu_warm)
        # use MPI to get the totals
        comm.Reduce([fourPiNujNu_warm, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        tmp = comm.bcast(tmp, root=0)
        fourPiNujNu_warm = np.copy(tmp)
        # t_stop = MPI.Wtime()
        # if rank == 0:
        #     print("Done! Took", (t_stop-t_start), "s", flush=True)

        hotInt = np.zeros_like(energy)
        for i in range(rank, energy.shape[0], size):
            hotInt[i] = simpson(fourPiNujNu_hot[:, i] * gvh, xh)
        tmp = np.zeros_like(hotInt)
        # use MPI to get the totals
        comm.Reduce([hotInt, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        tmp = comm.bcast(tmp, root=0)
        hotInt = np.copy(tmp)

        warmInt = np.zeros_like(energy)
        for i in range(rank, energy.shape[0], size):
            warmInt[i] = simpson(fourPiNujNu_warm[:, i] * gvw, xw) 
        tmp = np.zeros_like(warmInt)
        # use MPI to get the totals
        comm.Reduce([warmInt, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        tmp = comm.bcast(tmp, root=0)
        warmInt = np.copy(tmp)

        # erg/s/cm^2/deg^2/keV
        SB_hot += hotInt * ds_val 
        SB_warm += warmInt * ds_val
        t_stop = MPI.Wtime()
        if rank == 0:
            print(f"LOS: {los_val:.3f} kpc ({indx+1}/{LOSsample.shape[0]}) Done! Took {(t_stop-t_start):.4f} s", flush=True, end="\r")
    if rank == 0:
        print()
    return np.vstack((energy, SB_hot, SB_warm)).T
    
    

