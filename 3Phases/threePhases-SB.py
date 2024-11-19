#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:11:37 2023

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
import pickle
from typing import Optional
from misc.constants import kpc, pc
from observable.sb import SB_gen
from misc.template import unmodified_field, modified_field
from misc.HaloModel import HaloModel
from observable.disk_measures import Disk_profile
from astro_plasma import EmissionSpectrum, Ionization
from observable.CoordinateTrans import toGalC
from scipy.integrate import quad, simpson
from mpi4py import MPI

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

once = 0
spectrum_disk: Optional[np.ndarray] = None
calculate_ions = False
calculate_temp = True

metallicity = 0.3
redshift = 0.2
mode = "PIE"

def lnprob2D(x, y, params):
    xx, yy = np.meshgrid(x, y)
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    prob = np.array([np.log(fv[i]/(2*np.pi*sig[i,0]*sig[i,1])) - ( 
                                                 A[i]*(xx-xmed[i])**2 +  
                                                 C[i]*(yy-ymed[i])**2 +
                                                 B[i]*(xx-xmed[i])*(yy-ymed[i]) )
        for i in range(fv.shape[0])])
    return {"hot" : prob[0,:,:],
            "warm": prob[1,:,:],
            "cold": prob[2,:,:],
            }

if rank==0:
    with open("params_data.pickle", "rb") as file_obj:
        data = pickle.load(file_obj)
else:
    data = None
data = comm.bcast(data, root=0)

fV = data["fV"]
T_meds = data["T_meds"]
nH_meds = data["nH_meds"]
sigs = data["sigs"]
alphas = data["alphas"]
A,B,C = data["ABC"]
Temperature = data["Temperature"]
nH = data["nH"]
xi, yi = data["xy_meds"]
T_medV_u = data["T_medV_u"]
nH_medV_u = data["nH_medV_u"]

x = np.log(nH/nH_medV_u)
y = np.log(Temperature/T_medV_u)

nH_grd, T_grd = np.meshgrid(nH, Temperature)

if rank == 0:
    print(f"{'Calculating' if calculate_ions else 'Loading'} average electron, ion and hydrogen number densities", flush = True)
if calculate_ions or calculate_temp:
    pdf_vol_nHT = lnprob2D(x, y,
                          {"fv": fV,
                           "sig": sigs,
                           "median_nH": xi,
                           "median_T": yi,
                           "factors": [A,B,C]
                          })
    hot_pdf = np.copy(pdf_vol_nHT["hot"])/np.log(10)
    warm_pdf = np.copy(pdf_vol_nHT["warm"])/np.log(10)
    cold_pdf = np.copy(pdf_vol_nHT["cold"])/np.log(10)
    
    total_pdf = 10.**hot_pdf +10.**warm_pdf + 10.**cold_pdf
    if calculate_ions:
        num_dens = Ionization.interpolate_num_dens
        ne_local = np.zeros_like(nH_grd.flatten())
        ne_local[rank:ne_local.shape[0]:size] = num_dens(nH_grd.flatten()[rank:ne_local.shape[0]:size], 
                                                         T_grd.flatten()[rank:ne_local.shape[0]:size], 
                                                         metallicity, redshift, mode, "electron")
        tmp = np.zeros_like(ne_local)
        # use MPI to get the totals
        comm.Reduce([ne_local, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        ne_local = np.copy(tmp)

        ne_local = np.reshape(ne_local,nH_grd.shape)
        ne_global_avg = np.sum(ne_local * total_pdf * (x[1]-x[0])*(y[1]-y[0]))
        
        ni_local = np.zeros_like(ne_local)
        ni_local[rank:ni_local.shape[0]:size] = num_dens(nH_grd.flatten()[rank:ni_local.shape[0]:size], 
                                                         T_grd.flatten()[rank:ni_local.shape[0]:size], 
                                                         metallicity, redshift, mode, "ion")
        tmp = np.zeros_like(ni_local)
        # use MPI to get the totals
        comm.Reduce([ni_local, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        ni_local = np.copy(tmp)

        ni_local = np.reshape(ni_local,nH_grd.shape)
        ni_global_avg = np.sum(ni_local * total_pdf * (x[1]-x[0])*(y[1]-y[0]))
        
        nH_global_avg = np.sum(nH_grd * total_pdf * (x[1]-x[0])*(y[1]-y[0]))
    
        if rank==0:
            save = [ne_global_avg, ni_global_avg, nH_global_avg]
            np.savetxt(f"3p-ndens_{mode}.txt", save)
        else:
            save = None
        save = comm.bcast(save, root=0)
        if rank!=0:
            ne_global_avg, ni_global_avg, nH_global_avg = save
    else:
        ne_global_avg, ni_global_avg, nH_global_avg = np.loadtxt(f"3p-ndens_{mode}.txt")
    
    T_avg_global = np.sum(nH_grd * T_grd * total_pdf * (x[1]-x[0])*(y[1]-y[0]))/nH_global_avg
    if rank==0:
        np.savetxt(f"3p-temp_{mode}.txt", [T_avg_global])
else:
    if rank==0:
        T_avg_global = np.loadtxt(f"3p-temp_{mode}.txt")
    else:
        T_avg_global = None
    T_avg_global = comm.bcast(T_avg_global, root=0)

if rank==0:    
    print("Global averaged", flush=True)
    print(f"nH = {nH_global_avg:.3e}", flush=True)
    print(f"ni = {ni_global_avg:.3e}", flush=True)
    print(f"ne = {ne_global_avg:.3e}", flush=True)
    print(f"T = {T_avg_global:.3e}", flush=True)
    
M200 = 1.e12
halo = HaloModel(M200=M200)
rCGM = 1.1 * halo.r200 * (halo.UNIT_LENGTH / kpc)
r200 = halo.r200 * (halo.UNIT_LENGTH / kpc)
   
n_prof = lambda r, n1z, alpha: n1z * (1-alpha/3) * \
                               ((1-(r0/rCGM)**3)/(1-(r0/rCGM)**(3-alpha))) * \
                                   (r/rCGM)**(-alpha)
_block = False
    
def cgm_sb(L: float, B: float, rCGM: float, redshift:float, alpha:float, band: list[float, float]) -> float:
    disk = Disk_profile(redshift=redshift) # change this if you want different disk
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same
    band_cond = np.logical_and(energy>=band[0], energy<=band[1])
    
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
    LOSsample = np.linspace(0., integrateTill, Npoints)
    rad, phi, theta = toGalC(L, B, LOSsample)

    profile_los = n_prof(rad, nH_global_avg, alpha)
    condition = profile_los>1.0e-06 # (1.0e-03*nH_global_avg)

    ds_val = LOSsample[1] - LOSsample[0] # kpc

    spectrum = 0 # np.zeros_like(energy)
    for i in range(rank, radius.shape[0], size):
        if condition[i]:
            # spectrum += profile_los[i]
            spectrum += simpson((fourPiNujNu(profile_los[i], T_avg_global, 
                           metallicity, redshift, 
                           mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy)[band_cond], energy[band_cond]) # per sq deg
            # pass
    spectrum = spectrum * ds_val * kpc # convert to CGS
    tmp = np.array([0.,]) # np.zeros_like(spectrum)
    # use MPI to get the totals
    comm.Reduce([np.array([spectrum,]), MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                 op=MPI.SUM, root=0)
    comm.Barrier()
    tmp = comm.bcast(tmp, root=0)
    spectrum = tmp[0]
    return spectrum

def disk_sb(L: float, B: float, rCGM: float, redshift:float, band: list[float, float]) -> float:
    disk = Disk_profile(redshift=redshift) # change this if you want different disk
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same
    temperature: float = disk.TDisk
    metallicity: float = disk.metallicity
    redshift: float = disk.redshift
    mode: str = disk.mode
    band_cond = np.logical_and(energy>=band[0], energy<=band[1])
    
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
    if not(_block):
        profile_los = disk.nH(radius, height)
    else:
        profile_los = disk.nH_block(radius, height)
    condition = profile_los>9.9e-05

    ds_val = LOSsample[1:] - LOSsample[:-1] # kpc

    spectrum = 0 # np.zeros_like(energy)
    for i in range(rank, radius.shape[0], size):
        if condition[i]:
            # spectrum += profile_los[i]*ds_val[i]
            spectrum += simpson((fourPiNujNu(profile_los[i], 
                                              temperature, 
                                              metallicity, 
                                              redshift, mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy)[band_cond], energy[band_cond])*ds_val[i]
    spectrum = spectrum * kpc #convert to CGS
    tmp = np.array([0.,]) # np.zeros_like(spectrum)
    # use MPI to get the totals
    comm.Reduce([np.array([spectrum,]), MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                 op=MPI.SUM, root=0)
    comm.Barrier()
    tmp = comm.bcast(tmp, root=0)
    spectrum = tmp[0]
    return spectrum

r0 = 9.0 # kpc
rSG = 8.0 # kpc
Npoints = 1000
radius = np.logspace(np.log10(rSG), np.log10(rCGM), Npoints)  # kpc

def Integral_LOS(func, l, b, *args):
    _simpson = True
    if _simpson:
        f_val = np.zeros_like(radius) # evaluate the observable
        for i in range(rank, radius.shape[0], size):
            f_val[i] = func(radius[i], *args)
        tmp = np.zeros_like(radius)
        # use MPI to get the totals
        comm.Reduce([f_val, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                    op=MPI.SUM, root=0)
        comm.Barrier()
        tmp = comm.bcast(tmp, root=0)
        f_val = np.copy(tmp)
    s0 = rSG * np.sqrt(1.-( np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)) )**2)
    if _simpson:
        I = simpson(radius * f_val / np.sqrt(radius**2-s0**2), radius)* kpc  # f(n) CGS L0
    else:
        I = (quad(lambda r_val: r_val * func(r_val, *args) / np.sqrt(r_val**2-s0**2),
                  rSG, rCGM)[0])* kpc  # f(n) CGS L0
    if (l>=0 and l<=90) or (l>=270 and l<=360):
        if _simpson:
            eps = 1.0e-08
            radius_in = np.linspace(s0+eps, rSG, 50)  # kpc
            f_val_int = np.zeros_like(radius_in)
            for i in range(rank, radius_in.shape[0], size):
                f_val_int[i] = 2*func(radius_in[i], *args)
            tmp = np.zeros_like(radius_in)
            # use MPI to get the totals
            comm.Reduce([f_val_int, MPI.DOUBLE], [tmp, MPI.DOUBLE], 
                        op=MPI.SUM, root=0)
            comm.Barrier()
            tmp = comm.bcast(tmp, root=0)
            f_val_int = np.copy(tmp)
            I += (simpson(radius_in * f_val_int / np.sqrt(radius_in**2-s0**2), radius_in)* kpc) # f(n) CGS L0
        else:
            I += (quad(lambda r_val: 2*r_val * func(r_val, *args) / np.sqrt(r_val**2-s0**2),
                  s0, rSG)[0])* kpc  # f(n) CGS L0 The factor 2 is only true for spherical symmetry
    return I
 
# ---- eFEDS -----
l, b = [230, 30] # Ponti
if rank==0:
    print("Ponti", flush=True)

alphas = np.array([0,1.01,2.0])
bands = [
    [0.3,0.6], 
    [0.6,2.0],
    ]

def spectrum_band(nH_val: float, band: list[float, float], 
                  T_val: Optional=None,
                  met_val: Optional=None):
    fourPiNujNu = EmissionSpectrum.interpolate_spectrum
    energy = EmissionSpectrum._energy  # This remains the same
    condition = np.logical_and(energy>=band[0], energy<=band[1])
    emission = fourPiNujNu(nH_val, T_avg_global if T_val is None else T_val, 
                           metallicity if met_val is None else met_val, redshift, 
                           mode)[:, 1]/(4*np.pi)*(np.pi/180)**2/energy # per sq deg
    # print(nH_val, np.max(emission[condition]))
    return simpson(emission[condition], energy[condition])
'''
small = (90<l and l<270)
path1 = np.abs(rSG*np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b))*( 1 + np.sqrt(1.+((rCGM/rSG)**2-1)/(np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)))**2) ))
path2 = np.abs(rSG*np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b))*( 1 - np.sqrt(1.+((rCGM/rSG)**2-1)/(np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)))**2) ))
path_large = path1 if path1>path2 else path2
path_small = path1 if path1<path2 else path2

expectation = nH_global_avg * (path_small if small else path_large) * kpc
'''
if rank==0:
    print("Calculating spectrum...", flush=True)
    '''
    print("Paths: ", path1, path2, flush=True)
    print("Path chosen: ", (path_small if small else path_large), flush=True)
    print("Expect: ", expectation, flush=True)
    '''

CGM = np.zeros((len(alphas),len(bands)), dtype=np.float64)

for indx, alpha in enumerate(alphas):
    if rank==0:
        print("alpha = ", alpha, flush=True)    
    I = [cgm_sb(l, b, rCGM, redshift, alpha, band) for band in bands]
    CGM[indx, :] = np.array(I)
    if rank==0: 
        print("Sampling method", flush=True)
        print("bands: ", bands, flush=True)
        print(I, flush=True)  
comm.Barrier()


for indx, alpha in enumerate(alphas):
    if rank==0:
        print("alpha = ", alpha, flush=True)
    def f_val(rad, band: list[float, float]):
        nH_val = n_prof(rad, nH_global_avg, alpha)
        if nH_val<1.0e-06: # (1.0e-03*nH_global_avg):
            return 0.
        emission = spectrum_band(nH_val, band)
        # emission = nH_val if nH_val>9.9e-05 else 0.
        return emission 
    
    I = [Integral_LOS(f_val, l, b, band) for band in bands]
    CGM[indx, :] = np.array(I) 
    if rank==0:   
        print("Integral in r space", flush=True)
        print("bands: ", bands, flush=True)
        print(I, flush=True)

if rank==0: 
    print("disk", flush=True)
disk = Disk_profile(redshift=redshift)

'''
sgn = -1 if (90<l and l<270) else 1
path = 0.5*disk.z0/np.sqrt(1.-(np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)))**2)
if path > np.sqrt((disk.R0+sgn*rSG)**2 + (disk.z0/2)**2):
    path = np.abs( (disk.R0+sgn*rSG)/(np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b))) )

expectation = disk.nH0 * path * kpc
if rank==0:
    print("Expect: ", expectation, flush=True)
'''

def f_val(rad, band: list[float, float]):
    s1 = np.abs( rSG * np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)) * (1 + np.sqrt(1 + (rad**2-rSG**2)/(rSG* np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)))**2)) )
    s2 = np.abs( rSG * np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)) * (1 - np.sqrt(1 + (rad**2-rSG**2)/(rSG* np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)))**2)) )
    s_large = np.piecewise(s1, [s1>s2,], [s1, s2])
    s_small = np.piecewise(s1, [s1<s2,], [s1, s2])
    dist_los = np.copy(s_small) if (90<l and l<270) else np.copy(s_large)
    phi = np.deg2rad(l)
    theta = np.pi / 2 - np.deg2rad(b)
    xgc = dist_los * np.sin(theta) * np.cos(phi) - rSG
    ygc = dist_los * np.sin(theta) * np.sin(phi)
    zgc = dist_los * np.cos(theta)
    
    R_cyl, z_cyl = np.sqrt(xgc**2 + ygc**2), zgc

    if not(_block):
        nH_val = disk.nH(R_cyl, np.abs(z_cyl))
    else:
        nH_val = disk.nH_block(R_cyl, np.abs(z_cyl))
    if (nH_val>9.9e-05):
        emission = spectrum_band(nH_val, band, 
                                 disk.TDisk, disk.metallicity) 
    else:
        emission = 0
    # emission = nH_val
    return emission

I = [disk_sb(l, b, rCGM, redshift, band) for band in bands]
if rank==0: 
    print("Sampling method", flush=True)
    print("bands: ", bands, flush=True)
    print(I, flush=True)  
comm.Barrier()

if rank==0:
    print("total", flush=True)
for indx, alpha in enumerate(alphas):
    if rank==0:
        print("alpha = ", alpha, flush=True)
    total = CGM[indx, :] + np.array(I)
    if rank==0:
        print(total, flush=True)
comm.Barrier()

if rank==0: 
    print("disk", flush=True) 
I = [Integral_LOS(f_val, l, b, band) for band in bands]
if rank==0: 
    print("Integral in r space", flush=True)
    print("bands: ", bands, flush=True)
    print(I, flush=True)  

    print("total", flush=True)
for indx, alpha in enumerate(alphas):
    if rank==0:
        print("alpha = ", alpha, flush=True)
    total = CGM[indx, :] + np.array(I)
    if rank==0:
        print(total, flush=True)

