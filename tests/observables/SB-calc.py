# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:56:31 2022

@author: alankar
"""

import sys
import os

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
import pickle
from itertools import product
from typing import Optional
from misc.constants import kpc
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
from observable.sb import SB_gen
from misc.template import unmodified_field, modified_field
from observable.disk_measures import Disk_profile
from astro_plasma import EmissionSpectrum
from observable.CoordinateTrans import toGalC
from mpi4py import MPI

once = 0
spectrum_disk: Optional[np.ndarray] = None
only_disk_update = False
redshift = 0.2

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

def disk_sb(L: float, B: float, rCGM: float, redshift: float) -> np.ndarray:
    disk = Disk_profile(redshift=redshift) # change this if you want different disk
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same
    temperature: float = disk.TDisk
    metallicity: float = disk.metallicity
    redshift: float = disk.redshift
    mode: str = disk.mode
    
    SuntoGC = 8.0  # kpc
    costheta = np.cos(np.deg2rad(L)) * np.cos(np.deg2rad(B))
    root1 = np.abs(
            SuntoGC
            * costheta
            * (1 + np.sqrt(1 + (rCGM**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
        )
    root2 = np.abs(
            SuntoGC
            * costheta
            * (1 - np.sqrt(1 + (rCGM**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
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
            default=rCGM,
        )
    Npoints = 1000
    LOSsample = np.linspace(0., integrateTill, Npoints+1)
    rad, phi, theta = toGalC(L, B, 0.5*(LOSsample[1:]+LOSsample[:-1]) )
    height = np.abs(rad*np.cos(np.deg2rad(theta)))
    radius = np.abs(rad*np.sin(np.deg2rad(theta)))  # along disk
    profile_los = disk.nH(radius, height)
    condition = profile_los>9.9e-05

    ds_val = LOSsample[1:] - LOSsample[:-1] # kpc

    spectrum = np.zeros_like(energy)
    for i in range(rank, radius.shape[0], size):
        if condition[i]:
            spectrum += ((fourPiNujNu(profile_los[i], temperature, metallicity, redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy)*ds_val[i])
    spectrum = spectrum * kpc #convert to CGS
    tmp = np.zeros_like(spectrum)
    # use MPI to get the totals
    comm.Reduce([spectrum, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                 op=MPI.SUM, root=0)
    comm.Barrier()
    tmp = comm.bcast(tmp, root=0)
    spectrum = np.copy(tmp)

    return (energy, spectrum)

def spectrum(unmod: str, mod: str, ionization: str) -> None:
    # Ponti
    L = 230.0
    B = 30.0

    cutoff = 4.0
    TmedVW = 3.0e5
    sig = 0.3
    redshift = 0.0

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

    modified: modified_field = None
    if mod == "isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    global once
    global spectrum_disk
    if once == 0:
        if rank == 0:
            print("Calculating disk spectrum ...", end=" ", flush=True)
        t_start = MPI.Wtime()
        energy, spectrum_disk = disk_sb(L, B, unmodified.rCGM * unmodified.UNIT_LENGTH / kpc, redshift=redshift)
        t_stop = MPI.Wtime()
        once = 1
        if rank == 0:
            print("Done! Took", (t_stop-t_start) ,"s", flush=True)
            condition = np.logical_and(energy>=0.3, energy<=0.6)
            print("0.3-0.6 keV: ", np.trapz(spectrum_disk[condition], energy[condition]), "erg cm^-2 s^-1 deg^-2", flush=True)
            condition = np.logical_and(energy>=0.6, energy<=2.0)
            print("0.6-2.0 keV: ", np.trapz(spectrum_disk[condition], energy[condition]), "erg cm^-2 s^-1 deg^-2", flush=True)
    if rank == 0:
        print(unmod, mod, ionization, flush=True)
    spectrum: Optional[np.ndarray] = None
    if not(only_disk_update):
        if rank == 0:
            print("Calculating CGM spectrum ...", flush=True)
        t_start = MPI.Wtime()
        spectrum = SB_gen(mod, unmod, ionization, L, B, os.path.dirname(os.path.realpath(__file__)) )
        t_stop = MPI.Wtime()
        if rank == 0:
            print("Done! Took", (t_stop-t_start) ,"s", flush=True)
    else:
        data :Optional[dict] = None 
        if rank == 0:
            print("Reading CGM data ...", end=" ", flush=True)
            with open(f"SB_{unmod}_{mod}_{ionization}.pickle", "rb") as data_file:
                data = pickle.load(data_file)
            print("Done!", flush=True) 

        data = comm.bcast(data, root=0)    
        energy = np.array(data["energy"])
        sb_hot = np.array(data["sb_hot"])
        sb_warm = np.array(data["sb_warm"])
        spectrum = np.vstack((energy, sb_hot, sb_warm)).T

    if rank == 0:
        with open(f"SB_{unmod}_{mod}_{ionization}.pickle", "wb") as f:
            data = {
                "energy": spectrum[:, 0],
                "sb_hot": spectrum[:, 1],
                "sb_warm": spectrum[:, 2],
                "sb_disk": spectrum_disk,
                "rCGM": unmodified.rCGM * unmodified.UNIT_LENGTH / kpc,
            }
            pickle.dump(data, f)

if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor",]# "isobar"]
    ionization = ["PIE",]# "CIE"]

    for condition in product(unmod, mod, ionization):
        spectrum(*condition)
        
