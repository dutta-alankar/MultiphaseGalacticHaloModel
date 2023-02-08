#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:24:33 2023

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
import os

do_isothermal, do_isentropic =  True, True

npoints = 50

fig = plt.figure(figsize=(13,10), num=0)
ax1 = plt.gca()

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    os.system('mkdir -p ./isoth')
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 8.0
    THotM = TmedVH*np.exp(-sig**2/2)
    
    # PIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='PIE')
    redisProf  = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    

    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, npoints)    
    
    print('Generating profile ...')
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
        
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_global,
             color='tab:red', linestyle='-', linewidth=5, label=r'$\langle n^{(h)} \rangle_{g} (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:red', linestyle='--', linewidth=5, label=r'$\langle n^{(h)} \rangle (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_global,
             color='tab:blue', linestyle='-', linewidth=5, label=r'$\langle n^{(w)} \rangle_{g} (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_local,
             color='tab:blue', linestyle='--', linewidth=5, label=r'$\langle n^{(w)} \rangle (IC)$')
    # CIE
    # unmodified = IsothermalUnmodified(THot=THotM,
    #                       P0Tot=4580, alpha=1.9, sigmaTurb=60, 
    #                       M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
    #                       redshift=0.001, ionization='CIE')
    # redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
if(do_isothermal):
    os.system('mkdir -p ./isoth')
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 8.0
    THotM = TmedVH*np.exp(-sig**2/2)
    
    # PIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='PIE')
    redisProf  = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    

    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, npoints)    
    
    print('Generating profile ...')
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
        
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_global,
             color='tab:orange', linestyle='-', linewidth=5, label=r'$\langle n^{(h)} \rangle_{g} (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:orange', linestyle='--', linewidth=5, label=r'$\langle n^{(h)} \rangle (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_global,
             color='tab:cyan', linestyle='-', linewidth=5, label=r'$\langle n^{(w)} \rangle_{g} (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_local,
             color='tab:cyan', linestyle='--', linewidth=5, label=r'$\langle n^{(w)} \rangle (IB)$')


ax1.set_xscale('log')
ax1.set_yscale('log')    
ax1.set_ylim(1e-5,1e-2)
#plt.title('Isothermal density profile with isochoric and isobaric redistribution', size=25)
ax1.set_xlabel(r'$b/R_{vir}$', size=25)
ax1.set_ylabel(r'density $[cm^{-3}]$' ,size=25)
h1, l1 = ax1.get_legend_handles_labels()
leg = ax1.legend(h1, l1, loc='lower left', ncol=4, fancybox=True, fontsize=18)
ax1.tick_params(axis='both', which='major', length=12, width=3, labelsize=20)
ax1.tick_params(axis='both', which='minor', length=8, width=2, labelsize=20)
plt.grid()
plt.tight_layout()
leg.set_title("Isothermal profile with isochoric & isobaric redistribution", prop={'size':15})
plt.savefig('density_profiles_IT.png', transparent=True)
plt.show()
plt.close()
# ____________________________________________________________
# _________________ Isentropic profile _______________________

fig = plt.figure(figsize=(13,10), num=0)
ax1 = plt.gca()


if(do_isentropic):
    os.system('mkdir -p ./isent')
    nHrCGM = 1.1e-5
    TthrCGM = 2.4e5
    sigmaTurb = 60
    ZrCGM = 0.3
    TmedVW = 3.e5
    sig = 0.3
    cutoff = 8.0
    
    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, npoints)    
    
    print('Generating profile ...')
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_global,
             color='tab:green', linestyle='-', linewidth=5, label=r'$\langle n^{(h)} \rangle_{g} (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:green', linestyle='--', linewidth=5, label=r'$\langle n^{(h)} \rangle (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_global,
             color='tab:purple', linestyle='-', linewidth=5, label=r'$\langle n^{(w)} \rangle_{g} (IC)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_local,
             color='tab:purple', linestyle='--', linewidth=5, label=r'$\langle n^{(w)} \rangle (IC)$')
    # CIE
    # unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
    #                                   redshift=0.001, ionization='CIE')
    # redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)    
    
    # nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    # metallicity = redisProf.unmodified.metallicity

if(do_isentropic):
    os.system('mkdir -p ./isent')
    nHrCGM = 1.1e-5
    TthrCGM = 2.4e5
    sigmaTurb = 60
    ZrCGM = 0.3
    TmedVW = 3.e5
    sig = 0.3
    cutoff = 8.0
    
    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    redisProf = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, npoints)    
    
    print('Generating profile ...')
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_global,
             color='tab:pink', linestyle='-', linewidth=5, label=r'$\langle n^{(h)} \rangle_{g} (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:pink', linestyle='--', linewidth=5, label=r'$\langle n^{(h)} \rangle (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_global,
             color='tab:brown', linestyle='-', linewidth=5, label=r'$\langle n^{(w)} \rangle_{g} (IB)$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nwarm_local,
             color='tab:brown', linestyle='--', linewidth=5, label=r'$\langle n^{(w)} \rangle (IB)$')
             
ax1.set_xscale('log')
ax1.set_yscale('log')   
ax1.set_ylim(1e-6,1e-2) 
#plt.title('Isothermal density profile with isochoric and isobaric redistribution', size=25)
ax1.set_xlabel(r'$b/R_{vir}$', size=25)
ax1.set_ylabel(r'density $[cm^{-3}]$' ,size=25)
h1, l1 = ax1.get_legend_handles_labels()
leg = ax1.legend(h1, l1, loc='lower left', ncol=4, fancybox=True, fontsize=18)
ax1.tick_params(axis='both', which='major', length=12, width=3, labelsize=20)
ax1.tick_params(axis='both', which='minor', length=8, width=2, labelsize=20)
plt.grid()
plt.tight_layout()
leg.set_title("Isentropic profile with isochoric & isobaric redistribution", prop={'size':15})
plt.savefig('density_profiles_IE.png', transparent=True)
plt.show()
plt.close()             

