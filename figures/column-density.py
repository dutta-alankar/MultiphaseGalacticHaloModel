#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:38:22 2022

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
from misc.constants import kpc
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from observable.ColumnIon import ion_column
import matplotlib.pyplot as plt
import os

'''
#Prepare canvas for plot
fig, axs = plt.subplots(2, 2, figsize=(14,11))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.28, hspace=0.02) # set the spacing between axes
axs = np.array([ [plt.subplot(gs[0]), plt.subplot(gs[1]), ], 
                 [plt.subplot(gs[2]), plt.subplot(gs[3]), ] ])
'''
do_isothermal, do_isentropic = True, True

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    os.system('mkdir -p ./isoth')
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 6.0
    THotM = TmedVH*np.exp(-sig**2/2)
    
    b = np.linspace(9.0,250,200) #kpc
    
    # PIE
    print('PIE')
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    column = ion_column(mod_isochor)
    
    print('R_vir = %.2f kpc'%(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc))    
    print("Evaluating Column densities...")
    
    print("NOVI_PIE")
    NOVI_PIE    =  column.gen_column(b, element=8, ion=6)
    np.save('./isoth/NOVI_PIE.npy', np.vstack( (b, NOVI_PIE) ).T)
    
    print("NNV_PIE")
    NNV_PIE  =  column.gen_column(b, element=7, ion=5)
    np.save('./isoth/NNV_PIE.npy', np.vstack( (b, NNV_PIE) ).T)
    
    print("NOVII_PIE")
    NOVII_PIE  =  column.gen_column(b, element=8, ion=7)
    np.save('./isoth/NOVII_PIE.npy', np.vstack( (b, NOVII_PIE) ).T)
    
    print("NOVIII_PIE")
    NOVIII_PIE  =  column.gen_column(b, element=8, ion=8)
    np.save('./isoth/NOVIII_PIE.npy', np.vstack( (b, NOVIII_PIE) ).T)
    
    # ------------------------------------------------------------------------------
    # CIE
    print('CIE')
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    column = ion_column(mod_isochor)
    
    print("NOVI_CIE")
    NOVI_CIE    =  column.gen_column(b, element=8, ion=6)
    np.save('./isoth/NOVI_CIE.npy', np.vstack( (b, NOVI_CIE) ).T)
    
    print("NNV_CIE")
    NNV_CIE  =  column.gen_column(b, element=7, ion=5)
    np.save('./isoth/NNV_CIE.npy', np.vstack( (b, NNV_CIE) ).T)
    
    print("NOVII_CIE")
    NOVII_CIE  =  column.gen_column(b, element=8, ion=7)
    np.save('./isoth/NOVII_CIE.npy', np.vstack( (b, NOVII_CIE) ).T)
    
    print("NOVIII_CIE")
    NOVIII_CIE  =  column.gen_column(b, element=8, ion=8)
    np.save('./isoth/NOVIII_CIE.npy', np.vstack( (b, NOVIII_CIE) ).T)
    '''
    print("Generating plots...")
    obsData = np.loadtxt('NOVI-obs.txt')
    axs[0, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_PIE, 
               label=r'$N_{OVI}$ (PIE)', linewidth=5)
    
    axs[0, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_CIE, 
               label=r'$N_{OVI}$ (CIE)', linewidth=5)
    axs[0, 0].plot(obsData[:,0], obsData[:,1], 'o', 
               label=r'observed data', markersize=15)
    obsData = np.loadtxt('NNV-obs.txt')
    axs[0, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NNV_PIE, 
               label=r'$N_{NV}$ (PIE)', linewidth=5)
    axs[0, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NNV_CIE, 
               label=r'$N_{NV}$ (CIE)', linewidth=5)
    axs[0, 0].plot(obsData[:,0], obsData[:,1], 'v', 
               label=r'observed data', markersize=15)
    
    axs[0, 0].grid()
    axs[0, 0].set_ylabel(r'Column density $[{\rm cm}^{-2}]$', size=20 )
    axs[0, 0].get_xaxis().set_ticklabels([])
    axs[0, 0].set_xlim(xmin=5e-2, xmax=1.2)
    axs[0, 0].set_ylim(ymin=2e13, ymax=2.3e15)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=18, direction="in", pad=5, labelbottom=False, size=6)
    axs[0, 0].tick_params(axis='both', which='minor', labelsize=16, direction="in", pad=5, labelbottom=False, size=3)
    axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend(loc='lower left', framealpha=0.3, shadow=False, fancybox=True)
    
    
    axs[1, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_PIE, 
              label=r'$N_{OVII}$ (PIE)', linewidth=5)
    axs[1, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_CIE, 
              label=r'$N_{OVII}$ (CIE)', linewidth=5)
    axs[1, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVIII_PIE, 
                label=r'$N_{OVIII}$ (PIE)', linewidth=5)
    axs[1, 0].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVIII_CIE, 
              label=r'$N_{OVIII}$ (CIE)', linewidth=5)
    axs[1, 0].grid()
    axs[1, 0].set_xlabel(r'$\rm b/r_{200}$', size=20)
    axs[1, 0].set_ylabel(r'Column density $[{\rm cm}^{-2}]$', size=20 )
    axs[1, 0].set_xlim(xmin=5e-2, xmax=1.2)
    axs[1, 0].set_ylim(ymin=5e13, ymax=5e16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=18, direction="in", pad=5, size=6)
    axs[1, 0].tick_params(axis='both', which='minor', labelsize=16, direction="in", pad=5, size=3)
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].legend(loc='lower left', framealpha=0.3, shadow=False, fancybox=True) 
    '''
# ____________________________________________________________
# _________________ Isentropic profile _______________________

if(do_isentropic):
    os.system('mkdir -p ./isent')
    nHrCGM = 1.1e-5 # K
    TthrCGM = 2.4e5 # K
    sigmaTurb = 60 # km/s
    ZrCGM = 0.3
    TmedVW = 3.e5 # K
    sig = 0.3
    cutoff = 4.0
    
    b = np.linspace(9.0,250,200) #kpc
    
    # PIE
    print('PIE')
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    column = ion_column(mod_isochor)
    
    print('R_vir = %.2f kpc'%(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc))    
    print("Evaluating Column densities...")
    
    print("NOVI_PIE")
    NOVI_PIE    =  column.gen_column(b, element=8, ion=6)
    np.save('./isent/NOVI_PIE.npy', np.vstack( (b, NOVI_PIE) ).T)
    
    print("NNV_PIE")
    NNV_PIE  =  column.gen_column(b, element=7, ion=5)
    np.save('./isent/NNV_PIE.npy', np.vstack( (b, NNV_PIE) ).T)
    
    print("NOVII_PIE")
    NOVII_PIE  =  column.gen_column(b, element=8, ion=7)
    np.save('./isent/NOVII_PIE.npy', np.vstack( (b, NOVII_PIE) ).T)
    
    print("NOVIII_PIE")
    NOVIII_PIE  =  column.gen_column(b, element=8, ion=8)
    np.save('./isent/NOVIII_PIE.npy', np.vstack( (b, NOVIII_PIE) ).T)
    
    # ------------------------------------------------------------------------------
    # CIE
    print('CIE')
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    column = ion_column(mod_isochor)
    
    print("NOVI_CIE")
    NOVI_CIE    =  column.gen_column(b, element=8, ion=6)
    np.save('./isent/NOVI_CIE.npy', np.vstack( (b, NOVI_CIE) ).T)
    
    print("NNV_CIE")
    NNV_CIE  =  column.gen_column(b, element=7, ion=5)
    np.save('./isent/NNV_CIE.npy', np.vstack( (b, NNV_CIE) ).T)
    
    print("NOVII_CIE")
    NOVII_CIE  =  column.gen_column(b, element=8, ion=7)
    np.save('./isent/NOVII_CIE.npy', np.vstack( (b, NOVII_CIE) ).T)
    
    print("NOVIII_CIE")
    NOVIII_CIE  =  column.gen_column(b, element=8, ion=8)
    np.save('./isent/NOVIII_CIE.npy', np.vstack( (b, NOVIII_CIE) ).T)    
    '''
    axs[0, 1].grid()
    axs[0, 1].set_ylabel(r'Column density $[{\rm cm}^{-2}]$', size=20 )
    axs[0, 1].get_xaxis().set_ticklabels([])
    axs[0, 1].set_xlim(xmin=5e-2, xmax=1.2)
    axs[0, 1].set_ylim(ymin=2e13, ymax=2.3e15)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=18, direction="in", pad=5, labelbottom=False, size=6)
    axs[0, 1].tick_params(axis='both', which='minor', labelsize=16, direction="in", pad=5, labelbottom=False, size=3)
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].legend(loc='lower left', framealpha=0.3, shadow=False, fancybox=True)
    
    
    axs[1, 1].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_PIE, 
              label=r'$N_{OVII}$ (PIE)', linewidth=5)
    axs[1, 1].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_CIE, 
              label=r'$N_{OVII}$ (CIE)', linewidth=5)
    axs[1, 1].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVIII_PIE, 
               label=r'$N_{OVIII}$ (PIE)', linewidth=5)
    axs[1, 1].plot(b/(mod_isochor.unmodified.Halo.r200*mod_isochor.unmodified.Halo.UNIT_LENGTH/kpc), NOVIII_CIE, 
              label=r'$N_{OVIII}$ (CIE)', linewidth=5)
    axs[1, 1].grid()
    axs[1, 1].set_xlabel(r'$\rm b/r_{200}$', size=20)
    axs[1, 1].set_ylabel(r'Column density $[{\rm cm}^{-2}]$', size=20 )
    axs[1, 1].set_xlim(xmin=5e-2, xmax=1.2)
    axs[1, 1].set_ylim(ymin=5e13, ymax=5e16)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=18, direction="in", pad=5, size=6)
    axs[1, 1].tick_params(axis='both', which='minor', labelsize=16, direction="in", pad=5, size=3)
    axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend(loc='lower left', framealpha=0.3, shadow=False, fancybox=True) 
    '''
# plt.savefig('test.png', transparent=True)
# plt.show()
