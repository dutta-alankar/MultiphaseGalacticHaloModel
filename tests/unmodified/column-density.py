#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:02:21 2023

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
sys.path.append("../observables")
from astro_plasma import Ionization
import numpy as np
from typing import Optional
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from misc.template import unmodified_field
from scipy.interpolate import interp1d
from scipy import integrate
from mpi4py import MPI

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

load_ion = False
unmod = "isoth" # "isent"

ionization = "PIE"
redshift = 0.2
element = "OVI"

cutoff = 4.0  # Mukesh
TmedVW = 3.0e5
sig = 0.3
redshift = 0.2
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
elif unmod == "isent":
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
else:
    if rank==0:
        print(f"unmod: {unmod} is not supported!", flush=True)
    sys.exit(1)

pc = 3.0856775807e18
kpc = 1e3 * pc

if rank==0:
    print(f"Unmodified model: {unmod}", flush=True)
    # print(f"element: {element}", flush=True)
    print(f"r_CGM = {(unmodified.rCGM * unmodified.UNIT_LENGTH / kpc):.1f} kpc", flush=True)


b = np.linspace(9.0, 0.99*unmodified.rCGM * unmodified.UNIT_LENGTH / kpc, 200)  # kpc
Npts = 80
radius = np.linspace(
    unmodified.Halo.r0 * unmodified.Halo.UNIT_LENGTH / kpc,
    unmodified.rCGM * unmodified.UNIT_LENGTH / kpc,
    Npts,
)  # kpc

if not (load_ion):
    if rank==0:
        print("Calculating profile ...", end=" ", flush=True)
    rho, PTh, PNTh, PTurb, Ptot, nH_prof, mu = unmodified.ProfileGen(radius)
    Temperature_prof = unmodified.Temperature
    metallicity_prof = unmodified.metallicity
    if rank==0:
        t_stop = MPI.Wtime()
        print(f"Done! (took {(t_stop-t_start)} s)", flush=True)
        t_start = t_stop
else:
    if rank==0:
        data = np.loadtxt(f"./nIon_profile-{''.join(element.split())}-unmod_{unmod}_{ionization}.txt")
        comm.Bcast([data, MPI.DOUBLE], root=0)
    radius = data[:,0]
    nIon = data[:,1]


if not (load_ion):
    num_dens = Ionization.interpolate_num_dens
    if rank==0:
        np.savetxt("tmp.txt", np.vstack( (radius, nH_prof, Temperature_prof, metallicity_prof) ).T)
        print("Calculating ion density profile ... ", end=" ", flush=True)
    nIon = np.zeros_like(nH_prof)
    nIon[rank:nH_prof.shape[0]:size] = num_dens(nH_prof[rank:nH_prof.shape[0]:size], 
                                                Temperature_prof[rank:nH_prof.shape[0]:size], 
                                                metallicity_prof[rank:nH_prof.shape[0]:size], 
                                                redshift, mode=ionization, 
                                                element=element)
    # use MPI to get the totals
    _tmp = np.zeros_like(nH_prof)
    comm.Allreduce([nIon, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
    nIon = np.copy(_tmp)
    comm.Barrier()
    if rank==0:
        t_stop = MPI.Wtime()
        print(f"Done! (took {(t_stop-t_start)} s)", flush=True)
        t_start = t_stop
        np.savetxt(f"./nIon_profile-{''.join(element.split())}-unmod_{unmod}_{ionization}.txt", 
                   np.vstack((radius, nIon)).T)

if rank==0:
    log_nIon_prof = interp1d(radius, np.log10(nIon), fill_value="extrapolate")
    
    column_density = np.zeros_like(b)
    
    print(f"Calculating column density of {element} ...", end=" ", flush=True)
    epsilon = 1e-6
    for indx, impact in enumerate(b):
        column_density[indx] = (
                2
                * integrate.quad(
                    lambda r: (10.**log_nIon_prof(r)) * r / np.sqrt(r**2 - impact**2),
                    impact * (1 + epsilon),
                    unmodified.rCGM
                    * unmodified.UNIT_LENGTH
                    / kpc,
                )[0]
            )  # kpc cm^-3
        '''
        rad_vals = np.linspace(impact*(1+1.0e-06), 
                               np.sqrt((unmodified.rCGM * unmodified.UNIT_LENGTH / kpc)**2-impact**2),
                               50)
        condition = rad_vals>impact
        column_density[indx] = 2*kpc*np.trapz(
                                rad_vals[condition]*(10.**nIon_prof(rad_vals))[condition]\
                                    /np.sqrt(rad_vals[condition]**2-impact**2), 
                                rad_vals[condition])
        '''
    column_density = column_density*kpc # cm^-2
    t_stop = MPI.Wtime()
    print(f"Done! (took {(t_stop-t_start)} s)", flush=True)
    np.savetxt(f"./cdens_profile-{''.join(element.split())}-unmod_{unmod}_{ionization}.txt",
               np.vstack( (b/(unmodified.Halo.r200 * unmodified.Halo.UNIT_LENGTH / kpc), column_density) ).T)
MPI.Finalize()
