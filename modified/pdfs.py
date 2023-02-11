# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:01:08 2022

@author: Alankar
"""

import sys
sys.path.append('..')
import numpy as np
from scipy import interpolate
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
import matplotlib.pyplot as plt

do_isothermal, do_isentropic  = True, True
# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    sigH = sig
    sigW = sig
    cutoff = 8.0
    THotM = TmedVH*np.exp(-sig**2/2)
    
    radius = np.linspace(9.0,250,30) #kpc
    
    ionization = 'PIE'
   
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization=ionization)
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = mod_isochor.ProfileGen(radius)    
    
    radius, fvw, fmw, Tcut = radius[5], fvw[5], fmw[5], Tcut[5]
    fvw1 = fvw
    Tstart = 4.1
    Tstop  = 7.9
    Temp  = np.logspace(Tstart, Tstop, 1000)
    
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111)
    x = np.log(Temp/TmedVH)
    gvh = np.exp(-x**2/(2*sigH**2))/(sigH*np.sqrt(2*pi))
    xp = np.log(Temp/TmedVW)
    gvw = fvw*np.exp(-xp**2/(2*sigW**2))/(sigW*np.sqrt(2*pi))
            
    plt.vlines(np.log10(Tcut), 1e-3, 4.1, colors='black', linestyles='--', label=r'$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$'%cutoff, 
                         linewidth=5, zorder=20)
    plt.vlines(np.log10(TmedVH), 1e-3, 4.1, colors='tab:red', linestyles=':', label=r'$T_{med,V}^{(h)}$', 
                         linewidth=5, zorder=30)
    plt.vlines(np.log10(TmedVW), 1e-3, 4.1, colors='tab:blue', linestyles=':', label=r'$T_{med,V}^{(w)}$', 
                         linewidth=5, zorder=40)
    plt.semilogy(np.log10(Temp), gvh, color='tab:red', alpha=0.5, label='hot, unmodified', 
                         linewidth=5, zorder=6)        
    plt.semilogy(np.log10(Temp), np.piecewise(gvh,[Temp>=Tcut,],[lambda val:val, lambda val:0]), color='tab:red', label='hot, modified (IC)', 
                         linewidth=5, zorder=5)
    plt.semilogy(np.log10(Temp), gvw, color='tab:blue', label='warm (IC)', linestyle='--', 
                         linewidth=5, zorder=7)
    
    ############################################### Isobaric redistribution case
    radius = np.linspace(9.0,250,30) #kpc
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization=ionization)
    mod_isobar = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = mod_isobar.ProfileGen(radius)    
    
    radius, fvw, fmw, Tcut = radius[5], fvw[5], fmw[5], Tcut[5]
    
    gvw = fvw*np.exp(-xp**2/(2*sigW**2))/(sigW*np.sqrt(2*pi))
            
    plt.semilogy(np.log10(Temp), np.piecewise((1.-fvw)/fvw1*gvh,[Temp>=Tcut,],[lambda val:val, lambda val:0]), color='tab:orange', label='hot, modified (IB)', 
                         linewidth=5, zorder=5)
    plt.semilogy(np.log10(Temp), gvw, color='tab:cyan', label='warm (IB)', linestyle='--', 
                         linewidth=5, zorder=7)                  
    plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
    plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
    plt.grid()
    plt.title(r'$r = $%.1f kpc [isothermal with isochoric and isobaric modification] (PIE)'%(radius), size=20)
    plt.ylim(1e-3, 4.1)
    plt.xlim(5, 7)
    plt.ylabel(r'$T \mathscr{P}_V(T)$', size=25)
    plt.xlabel(r'$\log_{10} (T [K])$', size=25)
    # ax.yaxis.set_ticks_position('both')
    plt.legend(loc='upper right', prop={'size': 18}, framealpha=0.3, shadow=False, fancybox=True, bbox_to_anchor=(1.1, 1))
    plt.savefig('isothermal_isoch_isobar_PDF_%s.png'%ionization, transparent=True)
    plt.show() 

