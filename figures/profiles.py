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
import os

do_isothermal, do_isentropic = True, True

npoints = 20

fig = plt.figure(figsize=(13,10), num=0)
ax1 = plt.gca()
ax2 = ax1.twinx()

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    os.system('mkdir -p ./isoth')
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 6.0
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
             color='tab:red', linestyle='-', linewidth=5, label=r'$n^{(g)}_{hot, isoth}$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:red', linestyle='--', linewidth=5, label=r'$n^{(l)}_{hot, isoth}$')
    ax2.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, prs_hot/(nhot_local*kB),
             color='tab:orange', linestyle='-', linewidth=5, label=r'$T_{hot, isoth}$')
    
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, fvw,
             color='black', linestyle='-', linewidth=5, label=r'$f^{(w)}_{v, isoth}$')
    # CIE
    # unmodified = IsothermalUnmodified(THot=THotM,
    #                       P0Tot=4580, alpha=1.9, sigmaTurb=60, 
    #                       M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
    #                       redshift=0.001, ionization='CIE')
    # redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    

# ____________________________________________________________
# _________________ Isentropic profile _______________________

if(do_isentropic):
    os.system('mkdir -p ./isent')
    nHrCGM = 1.1e-5
    TthrCGM = 2.4e5
    sigmaTurb = 60
    ZrCGM = 0.3
    TmedVW = 3.e5
    sig = 0.3
    cutoff = 4.0
    
    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, npoints)    
    
    print('Generating profile ...')
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_global,
             color='tab:blue', linestyle='-', linewidth=5, label=r'$n^{(g)}_{hot, isent}$')
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, nhot_local,
             color='tab:blue', linestyle='--', linewidth=5, label=r'$n^{(l)}_{hot, isent}$')
    ax2.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, prs_hot/(nhot_local*kB),
             color='tab:green', linestyle='-', linewidth=5, label=r'$T_{hot, isent}$')
    
    ax1.plot(radius/redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, fvw,
             color='black', linestyle='--', linewidth=5, label=r'$f^{(w)}_{v, isent}$')
    # CIE
    # unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
    #                                   redshift=0.001, ionization='CIE')
    # redisProf = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)    
    
    # nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    # metallicity = redisProf.unmodified.metallicity

ax1.set_xscale('log')
ax1.set_yscale('log')    
ax2.set_yscale('log')

ax1.set_xlabel(r'$b/R_{vir}$', size=28)
ax1.set_ylabel(r'Model profiles' ,size=28)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1+h2, l1+l2, loc='lower left', ncol=4, fancybox=True, fontsize=25)
ax1.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
ax1.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
ax2.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
ax2.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# leg.set_title("Isothermal profile with isochoric redistribution", prop={'size':20})
# plt.savefig('./isoth/.png', transparent=True)
plt.show()
plt.close()