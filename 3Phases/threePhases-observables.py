# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:42:10 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import matplotlib
import h5py
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.patheffects as path_effects
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import simpson, trapezoid, quad
sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
sys.path.append("../tests/observables")

from obs_data import parse_cgm2, parse_magiicat, parse_cubs
from astro_plasma import Ionization
from misc.HaloModel import HaloModel
from parse_observation import observedColDens
from misc.constants import mp, mH, kpc, Xp, MSun
from scipy.special import hyp2f1, gamma as GAMMA
from observable.disk_measures import DiskDM, DiskEM
from mpi4py import MPI

calculate_ions = False

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

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
matplotlib.rcParams["axes.axisbelow"] = True
# plt.style.use("dark_background")

hyp2f1regularized = lambda a1, a2, b1, z: hyp2f1(a1, a2, b1, z) / GAMMA(b1)

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

halo_id = 110
try:
    file_hdf = h5py.File(f"./Illustris-TNG50-1/halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

Group_r200 = np.array(file_hdf["Group_r200"])
Group_M200 = np.array(file_hdf["Group_M200"])

print("Debug", Group_r200)

mode = "PIE"
metallicity = 0.3
redshift = np.array(file_hdf["Redshift"])

if rank == 0:
    print(f"{metallicity, redshift, mode = }", flush=True)

XH = 0.76
gamma = 5 / 3.
yr = 365 * 24 * 60**2
Msun = 1.989e33
h = 0.6774

a = 1 / (1 + redshift)
ckpc = kpc / a

# M200 = Group_M200 * h**-1 * 1.0e+10 # Msun
# halo = HaloModel(M200=M200)
r200 = Group_r200 * (ckpc/h)/ kpc # halo.r200 * (halo.UNIT_LENGTH / kpc)
rCGM = 1.1 * r200

if rank == 0:
    print(f"r_vir = {r200:.1f} kpc, r_CGM = {rCGM:.1f} kpc", flush=True)

if rank==0:
    with open("params_data.pickle", "rb") as file_obj:
        data = pickle.load(file_obj)
else:
    data = None
data = comm.bcast(data, root=0)

# print("PDF data read: ", data)

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

if rank==0:
    print("Problem size: ", (nH.shape[0], Temperature.shape[0]), flush=True)

x = np.log(nH/nH_medV_u)
y = np.log(Temperature/T_medV_u)

# This volume fraction square is because global quantities come up calculation of column density
# On the other hand, P = n kB T , is where n is local
# One volume fraction comes from the pdf while the other is from local to global density conversion

if calculate_ions:
    if rank==0:
        print("Calculating the PDF from the parameters read in from disk .... ", 
              end="", flush=True)
    pdf_vol_nHT = lnprob2D(x, y,
                          {"fv": fV,
                           "sig": sigs,
                           "median_nH": xi,
                           "median_T": yi,
                           "factors": [A,B,C]
                          })
    hot_pdf = np.copy(pdf_vol_nHT["hot"])/np.log(10) # ln(PDF) -> log_10(PDF) in xy space
    warm_pdf = np.copy(pdf_vol_nHT["warm"])/np.log(10)
    cold_pdf = np.copy(pdf_vol_nHT["cold"])/np.log(10)
    
    total_pdf = 10.**hot_pdf + 10.**warm_pdf + 10.**cold_pdf
    if rank==0:
        print("Done!", flush=True)
    nH_grd, T_grd = np.meshgrid(nH, Temperature)

def nIon_global_avg(element=None, part_type=None):
    comm.Barrier()
    t_start = MPI.Wtime()
    if element is not None and part_type is not None:
        if rank==0:
            print("nIon_global_avg: invalid input!", flush=True)
        sys.exit(1)
    nIon = Ionization.interpolate_num_dens
    nH_this_proc = nH_grd.flatten()[rank : nH_grd.flatten().shape[0] : size]
    temperature_this_proc = T_grd.flatten()[rank : T_grd.flatten().shape[0] : size]
    nIon_local_this_proc = np.zeros_like(nH_grd.flatten())
    
    if element is None:
        if rank==0:
            print(f"{'Calculating' if calculate_ions else 'Loading'} average {part_type} density ... ", 
                  end="", flush=True)
        nIon_local_this_proc[rank : nH_grd.flatten().shape[0] : size] = nIon(
            nH_this_proc, temperature_this_proc,
            metallicity, redshift, mode, part_type = part_type,
            )
    else:
        if rank==0:
            print(f"{'Calculating' if calculate_ions else 'Loading'} average {element} density ... ", 
                  end="", flush=True)
        nIon_local_this_proc[rank : nH_grd.flatten().shape[0] : size] = nIon(
            nH_this_proc, temperature_this_proc,
            metallicity, redshift, mode, element=element,
            )
    comm.Barrier()

    nIon_local = np.zeros_like(nH_grd.flatten())
    # use MPI to get the totals
    comm.Reduce([nIon_local_this_proc, MPI.DOUBLE], [nIon_local, MPI.DOUBLE], 
                op=MPI.SUM, root=0)
    comm.Barrier()
    if rank==0:
        nIon_local = np.reshape(nIon_local, nH_grd.shape)
        nIOn = nIon_local * total_pdf * (x[1]-x[0])*(y[1]-y[0])
        nIon_g_avg_xy = np.sum(nIOn)
    else:
        nIon_g_avg_xy = None
    nIon_g_avg_xy = comm.bcast(nIon_g_avg_xy, root=0)
    comm.Barrier()
    if rank==0:
        print("Done!", flush=True)
    t_diff = MPI.Wtime() - t_start
    if rank == 0:
        print(f"Elapsed: {t_diff} s", flush=True)
    return nIon_g_avg_xy

if calculate_ions:
    # num_dens = Ionization.interpolate_num_dens
    
    # ne_local = np.array(
    #     [
    #         num_dens(nH_grd.flatten()[i], T_val, metallicity, redshift, mode, "electron")
    #         for i, T_val in enumerate(T_grd.flatten())
    #     ]
    # )
    # ne_local = np.reshape(ne_local,nH_grd.shape)
    ne_global_avg = nIon_global_avg(part_type="electron")
    
    # ni_local = np.array(
    #     [
    #         num_dens(nH_grd.flatten()[i], T_val, metallicity, redshift, mode, "ion")
    #         for i, T_val in enumerate(T_grd.flatten())
    #     ]
    # )
    # ni_local = np.reshape(ni_local,nH_grd.shape)
    # ni_global_avg = np.sum(ni_local * total_pdf * (x[1]-x[0])*(y[1]-y[0]))
    
    ni_global_avg = nIon_global_avg(part_type="ion")
    nH_global_avg = np.sum(nH_grd * total_pdf * (x[1]-x[0])*(y[1]-y[0]))

    save = [ne_global_avg, ni_global_avg, nH_global_avg]
    if rank==0:
        np.savetxt(f"3p-ndens_{mode}.txt", save)
else:
    if rank==0:
        ne_global_avg, ni_global_avg, nH_global_avg = np.loadtxt(f"3p-ndens_{mode}.txt")
    else:
        ne_global_avg, ni_global_avg, nH_global_avg = None, None, None
    ne_global_avg = comm.bcast(ne_global_avg, root=0)
    ni_global_avg = comm.bcast(ni_global_avg, root=0)
    nH_global_avg = comm.bcast(nH_global_avg, root=0)
    if rank==0:
        print("Loading from disk complete!", flush=True)

if rank==0:    
    print("Global averaged", flush=True)
    print(f"nH = {nH_global_avg}", flush=True)
    print(f"ni = {ni_global_avg}", flush=True)
    print(f"ne = {ne_global_avg}", flush=True)
    XH = 0.76
    print("M_CGM = %e MSun"%(4*np.pi/3*(rCGM*kpc)**3*nH_global_avg*mH/MSun/XH),
                        flush=True)

b = np.linspace(9.0, 1.05 * r200, 200)  # kpc
column_length = 2 * np.sqrt(rCGM**2 - b**2)

def IonColumn(element, ylim=None, fignum=None, color=None):
    if rank==0:
        if fignum is None:
            plt.figure(figsize=(13, 10))
        else:
            plt.figure(num=fignum, figsize=(13, 10))
        if color == None:
            color = "tab:blue"

    observation = observedColDens()

    if calculate_ions:
        nIon0 = nIon_global_avg(element="".join(element.split()))
        if rank==0:
            np.savetxt(f"nIon0-{element}_{mode}.txt", [nIon0])
    else:
        if rank==0:
            nIon0 = np.loadtxt(f"nIon0-{element}_{mode}.txt")
        else:
            nIon0 = None
        nIon0 = comm.bcast(nIon0, root=0)
    comm.Barrier()
    alpha = np.array([0,1.01,2.0])
    linestyles = np.array(["-","--",":"])
    if rank==0:
        print(f"element: {element}, n0: {nIon0}", flush=True)
    for indx, val in enumerate(alpha):
      factor_hype = GAMMA((val-1)/2)*((np.sqrt(np.pi)*(b/rCGM)**(1-val)/GAMMA(val/2))-hyp2f1regularized(0.5,(val-1)/2,(val+1)/2,(b/rCGM)**2))
      factor_den = (1.-val/3.) * (1.-(b[0]/rCGM)**3.) / (1.-(b[0]/rCGM)**(3.-val))

      NIon_prof = nIon0 * rCGM * kpc * factor_hype * factor_den
      if rank==0:
          print("alpha = ", val, flush=True)
          print(f"min: {np.min(NIon_prof):.4e}", flush=True)
          print(f"max: {np.max(NIon_prof):.4e}", flush=True)

          plt.plot(
            np.hstack([b / r200, (b[-1]/r200*(1.+1.0e-03))]),
            np.hstack([NIon_prof, [-np.inf]]),
            color=color,
            # label=r"$\rm N_{%s}, \alpha=%d$" % ("".join(element.split()),val),
            linewidth=5,
            linestyle=linestyles[indx],
            )
    if rank==0:   
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

        yerr = np.log(10) * e_coldens_detect * 10.0**coldens_detect
        plt.errorbar(
            impact_select_detect / rvir_select_detect,
            10.0**coldens_detect,
            yerr=yerr,
            fmt="o",
            color=color,
            # label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
            markersize=10,
        )
        plt.plot(
            impact_select_min / rvir_select_min,
            10.0**coldens_min,
            "^",
            color=color,
            markersize=10,
        )
        plt.plot(
            impact_select_max / rvir_select_max,
            10.0**coldens_max,
            "v",
            color=color,
            markersize=10,
        )

        if element == "Mg II":
            impact_magiicat, col_dens_magiicat, col_dens_err_magiicat = parse_magiicat.parse_col_dens_magiicat()
            condition = np.logical_and(col_dens_err_magiicat!=-1, col_dens_err_magiicat!= 1)
            cutoff_err = np.logical_and(condition, col_dens_err_magiicat<0.4)
            yerr = np.log(10) * col_dens_err_magiicat * 10.0**col_dens_magiicat
            
            plt.errorbar(
                impact_magiicat[cutoff_err],
                10.0**col_dens_magiicat[cutoff_err],
                yerr=yerr[cutoff_err],
                fmt="o",
                color=color,
               # label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
                markersize=10,
                markerfacecolor="None",
            )
            plt.plot(
                impact_magiicat[np.logical_not(condition)],
                10.0**(col_dens_magiicat[np.logical_not(condition)]),
                "v",
                color=color,
                markersize=10,
                markerfacecolor="None",
            )
            
            impact_cubs, col_dens_cubs, col_dens_err_cubs = parse_cubs.parse_col_dens_cubs("MgII")
            yerr = np.log(10) * col_dens_err_cubs * 10.0**col_dens_cubs
            ebar = plt.errorbar(
                impact_cubs,
                10.0**col_dens_cubs,
                yerr=yerr,
                fmt="o",
                color=color,
                # label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
                markersize=10,
                # markerfacecolor="None",
                markeredgecolor = "black",
                zorder=200,
                # hatch="/",
            )
            # taken from https://stackoverflow.com/questions/36173774/border-on-errorbars-in-matplotlib-python
            # ebar[1][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
            #                             path_effects.Normal()])
            # ebar[1][1].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
            #                            path_effects.Normal()])
            ebar[2][0].set_path_effects([path_effects.withStroke(linewidth=6, foreground='black', capstyle="round"),
                                        path_effects.Normal()])

        if element == "O VI":
            # CGM^2 obs
            impact_cgm2, col_dens_cgm2, col_dens_err_cgm2 = parse_cgm2.parse_col_dens_cgm2()
            condition = np.logical_and(col_dens_err_cgm2==-1, impact_cgm2<1.2)
            plt.plot(
                impact_cgm2[condition],
                10.0**(col_dens_cgm2[condition]),
                "v",
                color=color,
                markersize=10,
                markerfacecolor="None",
            )
            condition = np.logical_and(col_dens_err_cgm2!=-1, impact_cgm2<1.2)
            plt.plot(
                impact_cgm2[condition],
                10.0**(col_dens_cgm2[condition]),
                "o",
                color=color,
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
                color=color,
                # label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
                markersize=10,
                # markerfacecolor="None",
                markeredgecolor = "black",
                zorder=200,
                # hatch="/",
            )
            # taken from https://stackoverflow.com/questions/36173774/border-on-errorbars-in-matplotlib-python
            # ebar[1][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
            #                             path_effects.Normal()])
            # ebar[1][1].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
            #                            path_effects.Normal()])
            ebar[2][0].set_path_effects([path_effects.withStroke(linewidth=6, foreground='black', capstyle="round"),
                                        path_effects.Normal()])
            plt.plot(
                impact_cubs[np.logical_not(condition)],
                10.0**(col_dens_cubs[np.logical_not(condition)]),
                "v",
                color=color,
                markersize=10,
                markeredgecolor="black",
            )
        if element == "O VII":
            # obs
            NOVII_obs = 15.68
            NOVII_err = 0.27
            yerr = np.log(10) * NOVII_err * 10.0**NOVII_obs
            plt.axhspan(
                2 * 10.0**NOVII_obs,
                2 * (10.0**NOVII_obs - yerr),
                color="gray",
                alpha=0.2,
                zorder=0,
            )
            plt.ylim(ymin=2e13)
            #print("{:e}".format(2 * 10.0**NOVII_obs),"{:e}".format( 2 * (10.0**NOVII_obs - yerr)))
            
        if element == "O VIII":
            # obs
            NOVII_obs = 15.68
            NOVII_err = 0.27
            NOVIII_obs = NOVII_obs - np.log10(4)
            NOVIII_err = NOVII_err - np.log10(4)
            yerr = np.log(10) * NOVIII_err * 10.0**NOVIII_obs
            plt.axhspan(
                2 * 10.0**NOVIII_obs,
                2 * (10.0**NOVIII_obs - yerr),
                color="gray",
                alpha=0.2,
                zorder=0,
            )
            plt.ylim(ymin=2e13)
            #print("{:e}".format(2 * 10.0**NOVIII_obs), "{:e}".format(2 * (10.0**NOVIII_obs - yerr)))
            
        plt.xscale("log")
        plt.yscale("log")
        if ylim != None:
            plt.ylim(*ylim)
        plt.xlim(6e-2, 1.3)
        plt.xlabel(r"Impact parameter b [$r_{\rm vir}$]", size=28)
        plt.ylabel(r"Column density [$\rm cm^{-2}$]", size=28)
    
        plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
        plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
        plt.grid()
        plt.tight_layout()
        # set the linewidth of each legend object
        # for legobj in leg.legendHandles:
        # leg.set_title("Column density predicted by three phase model",prop={'size':20})
        if fignum == None:
            plt.legend(loc="lower left", ncol=1, fancybox=True, fontsize=25)
            plt.savefig(
                f"./figures/observations/N_%s-3p_{mode}.png" % ("".join(element.split()),),
                transparent=False,
            )
            # plt.show()
            plt.close()

    return nIon0

if rank==0:
    os.makedirs("./figures/observations/", exist_ok=True)
comm.Barrier()

# --------------------- DM and EM ------------------
if rank==0:
    print("Generating EM/DM for external galaxies", flush=True)
      
    DM = ne_global_avg * column_length * 1e3  # cm^-3 pc
    EM = ne_global_avg * nH_global_avg * column_length * 1e3  # cm^-6 pc
    
    # Upper limit
    print(f"Max possible DM [cm^-3 pc] (1-zone-constant): {(2 * ne_global_avg * rCGM * 1.e3):.3e}", flush=True)
    print(f"Max possible EM [cm^-6 pc] (1-zone-constant): {(2 * ne_global_avg * nH_global_avg * rCGM * 1.e3):.3e}", flush=True)

    # DM
    plt.figure(figsize=(13, 10))
    alphas = np.array([0,1.01,2.0])
    linestyles = np.array(["-","--",":"])
        
    for indx, val in enumerate(alphas):
        factor_hype = GAMMA((val-1)/2)*((np.sqrt(np.pi)*(b/rCGM)**(1-val)/GAMMA(val/2))-hyp2f1regularized(0.5,(val-1)/2,(val+1)/2,(b/rCGM)**2))
        factor_den = (1.-val/3.) * (1.-(b[0]/rCGM)**3.) / (1.-(b[0]/rCGM)**(3.-val))
        DM_prof = ne_global_avg * rCGM * factor_hype * factor_den *1e3 # cm^-3 pc
        
        plt.plot(np.hstack([b / r200, (b[-1]/r200*(1.+1.0e-03))]),
                 np.hstack([DM_prof, [-np.inf]]),
                 color="firebrick", 
                 label=r"$\rm\alpha=%d$" % val,
                 linewidth=5,
                 linestyle=linestyles[indx],
                 )
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(4.0, 60.0)
    # plt.xlim(6e-2, 1.2)
    plt.xlabel(r"Impact parameter $b/r_{\rm vir}$", size=28)
    plt.ylabel(r"DM [$cm^{-3} pc$]", size=28)
    plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
    plt.tight_layout()
    # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    # leg.set_title("Column density predicted by three phase model",prop={'size':20})
    plt.legend(loc="lower left", ncol=1, fancybox=True, fontsize=25)
    plt.savefig(f"./figures/observations/DM-3p_{mode}.png", transparent=False)
    # plt.show()
    plt.close()
        
    # EM
    plt.figure(figsize=(13, 10))
    alphas = np.array([0,1.01,2.0])
    linestyles = np.array(["-","--",":"])
        
    for indx, val in enumerate(alphas):
        factor_hype = GAMMA((val-1)/2)*((np.sqrt(np.pi)*(b/rCGM)**(1-val)/GAMMA(val/2))-hyp2f1regularized(0.5,(val-1)/2,(val+1)/2,(b/rCGM)**2))
        factor_den = (1.-val/3.) * (1.-(b[0]/rCGM)**3.) / (1.-(b[0]/rCGM)**(3.-val))
        EM_prof = ne_global_avg * nH_global_avg * rCGM * factor_hype * factor_den *1e3 # cm^-6 pc
        
        plt.plot(np.hstack([b / r200, (b[-1]/r200*(1.+1.0e-03))]),
                 np.hstack([EM_prof, [-np.inf]]),
                 color="firebrick", 
                 label=r"$\rm\alpha=%d$" % val,
                 linewidth=5,
                 linestyle=linestyles[indx],
                 )
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(4.0, 60.0)
    # plt.xlim(6e-2, 1.2)
    plt.xlabel(r"Impact parameter $b/r_{\rm vir}$", size=28)
    plt.ylabel(r"EM [$cm^{-6} pc$]", size=28)
    plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
    plt.tight_layout()
    # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    # leg.set_title("Column density predicted by three phase model",prop={'size':20})
    plt.legend(loc="lower left", ncol=1, fancybox=True, fontsize=25)
    plt.savefig(f"./figures/observations/EM-3p_{mode}.png", transparent=False)
    # plt.show()
    plt.close()

    # print(f"{'Calculating' if calculate_ions else 'Loading'} ion columns", flush=True)
'''
# ---------- Individual Ions -------------------------
element = "H I"
IonColumn(element)

element = "Mg II"
IonColumn(element)

element = "Si IV"
IonColumn(element)

element = "S III"
IonColumn(element)

element = "N V"
IonColumn(element)

element = "C III"
IonColumn(element)

element = "C II"
IonColumn(element)

element = "O VIII"
IonColumn(element)

element = "O VII"
IonColumn(element)

element = "O VI"
IonColumn(element, ylim=(10**14.0, 10.0**15.3))
'''

# all together
fignum = 100
if rank == 0:
    fig = plt.figure(figsize=(13, 10), num=fignum)
    # print('Figures: ', plt.get_fignums())
element = "O VI"
n1zOVI  = IonColumn(element, fignum=fignum, color="coral")
if rank == 0:
    legend_elements = [Patch(facecolor="coral", edgecolor="coral",
                             label="".join(element.split())) ]

element = "Mg II"
n1zMgII = IonColumn(element, fignum=fignum, color="lightseagreen")
if rank == 0:
    legend_elements.append(Patch(facecolor="lightseagreen", edgecolor="lightseagreen",
                             label="".join(element.split())) )
    
    leg1 = plt.gca().legend(handles=legend_elements, loc='lower center',
                     fancybox=True, fontsize=24, framealpha=0.1, ncol=2,
                     bbox_to_anchor=(0.24,0.01))
    legend_elements = [Line2D([0], [0], color='black', 
                              linewidth=5, 
                              linestyle=linestyles[i],
                              label=r'$\alpha=%d$'%alphas[i]) 
                       for i in range(len(linestyles))]
    leg2 = plt.gca().legend(handles=legend_elements, loc='lower center',
                     fancybox=True, fontsize=24, framealpha=0.1, ncol=4,
                     bbox_to_anchor=(0.37,0.07)
                     )
    # dummy
    dum1 = plt.errorbar([np.inf],
                [-np.inf],
                yerr=[1.0e-06],
                fmt="o",
                color="black",
                label=r"Observations",
                markersize=10,)
    dum2, = plt.plot([np.inf],
                [-np.inf],
                "v",
                color="black",
                label=r"Observations",
                markersize=10,)
    dum3, = plt.plot([np.inf],
                [-np.inf],
                "^",
                color="black",
                label=r"Observations",
                markersize=10,)
    
    plt.legend([(dum1, dum2, dum3)], ['Observations'], numpoints=1,
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  loc='lower center',
                  framealpha=0.1,
                  fancybox=False, fontsize=24,
                  bbox_to_anchor = (0.6,0.01))
    
    # plt.legend(loc="lower left", ncol=4, 
    #            fancybox=True, fontsize=18)
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)
    
    plt.ylim(5e10, 2e15)
    
    #plt.grid()
    plt.savefig(f"./figures/observations/N_OVI+MgII-3p_{mode}.png", transparent=False)
    # plt.show()
    plt.close()

r0 = 9.0 # kpc
rSG = 8.0 # kpc
radius = np.linspace(rSG, 1.05 * rCGM, 200)  # kpc
n_prof = lambda r, n1z, alpha: n1z * (1-alpha/3) * \
                               ((1-(r0/rCGM)**3)/(1-(r0/rCGM)**(3-alpha))) * \
                                   (r/rCGM)**(-alpha)

def Integral_LOS(func, l, b):
    # f_val = func(radius)
    s0 = rSG * np.sqrt(1-( np.cos(np.deg2rad(l))*np.cos(np.deg2rad(b)) )**2)
    # I = np.trapz(radius * f_val / np.sqrt(radius**2-s0**2), radius)* 1.0e+03  # f(n) pc
    I = (quad(lambda rp: rp*func(rp)/np.sqrt(rp**2-s0**2),
              rSG, 1.05 * rCGM)[0])*1.0e+03 # f(n) pc
    if (l>=0 and l<=90) or (l>=270 and l<=360): # passes through the inner sphere
        # radius_in = np.linspace(s0, rSG, 50)  # kpc
        # f_val_int = func(radius_in)
        # f_val_int = 2 * n_prof(radius_in, ne_global_avg, alpha)
        # I += (np.trapz(radius_in * f_val_int / np.sqrt(radius_in**2-s0**2), radius_in)* 1.0e+03) # f(n) pc
        I += (quad(lambda rp: 2*rp*func(rp)/np.sqrt(rp**2-s0**2),
                   s0, rSG)[0])*1.0e+03 # f(n) pc 
    return I
        
# ---- eFEDS -----
l, b = [230, 30] # Ponti
if rank == 0:
    print("Emission measure Ponti eFEDS", flush=True)

alphas = np.array([0,1.01,2.0])
linestyles = np.array(["-","--",":"]) 

for alpha in alphas:
    f_val = lambda rad: n_prof(rad, nH_global_avg, alpha) * n_prof(rad, ne_global_avg, alpha)
    I = Integral_LOS(f_val, l, b) # cm^-6 pc
    if rank == 0:
        print("alpha = ", alpha, flush=True)
        print(f"Ponti: {I:.3e}", flush=True)

disk = DiskEM(rvir=r200)
disk.set_disk(redshift=redshift, metallicity=1.0)
disk_em = disk.make_map(l, b, showProgress=False)
if rank==0:
    print(f"Disk EM : {disk_em:.3e}", flush=True)

# ---- FRB -----
l, b = [142, 41] # Bhardwaj
if rank == 0:
    print("Dispersion measure Bhardwaj FRB", flush=True)

for alpha in alphas:
    f_val = lambda rad: n_prof(rad, ne_global_avg, alpha) 
    I = Integral_LOS(f_val, l, b)
    if rank==0:
        print("alpha = ", alpha, flush=True)
        print(f"Bhardwaj: {I:.3e}", flush=True)

disk = DiskDM(rvir=r200)
disk.set_disk(redshift=redshift, metallicity=1.0)
disk_dm = disk.make_map(l, b, showProgress=False)
if rank==0:
    print(f"Disk DM : {disk_dm:.3e}", flush=True)
file_hdf.close()
MPI.Finalize()
