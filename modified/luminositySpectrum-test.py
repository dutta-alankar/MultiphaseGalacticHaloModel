#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:56:31 2022

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
from scipy import interpolate
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from observable.EmissionSpectrum import LuminositySpectrumGen
import matplotlib.pyplot as plt

do_isothermal, do_isentropic = False, True

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 8
    THotM = TmedVH*np.exp(-sig**2/2)
    
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12)
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVH, TmedVW, sig, cutoff)
        
    spectrum   =  LuminositySpectrumGen(mod_isochor, redshift=0.001)

    
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111)
    plt.loglog(spectrum[:,0], spectrum[:,1], 
               label=r'isothermal with isochoric modification', linewidth=2)
    plt.grid()
    ax.set_ylim(ymin=1e29, ymax=1e44)
    ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
    ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
    ax.set_ylabel(r'Luminosity [$erg/s/keV$]', size=28, color='black') 
    ax.set_xlabel(r'E [keV]', size=28, color='black')
    ax.set_xlim(xmin=5e-3, xmax=5e1)
    plt.legend(loc='lower left', prop={'size': 20}, framealpha=0.3, shadow=False, fancybox=True) #, bbox_to_anchor=(1.1, 1))
    plt.savefig('luminosity_isoTh_with_Isochor.png', transparent=True)
    plt.show()

# ____________________________________________________________
# _________________ Isentropic profile _______________________

if(do_isentropic):
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 0.001
    THotM = TmedVH*np.exp(-sig**2/2)
    
    unmodified = IsentropicUnmodified()
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVH, TmedVW, sig, cutoff)
    
    spectrum   =  LuminositySpectrumGen(mod_isochor, redshift=0.001)

    
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111)
    plt.loglog(spectrum[:,0], spectrum[:,1], 
               label=r'isentropic with isochoric modification', linewidth=2)
    plt.grid()
    ax.set_ylim(ymin=1e29, ymax=1e44)
    ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
    ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
    ax.set_ylabel(r'Luminosity [$erg/s/keV$]', size=28, color='black') 
    ax.set_xlabel(r'E [keV]', size=28, color='black')
    ax.set_xlim(xmin=5e-3, xmax=2e1)
    plt.legend(loc='lower left', prop={'size': 20}, framealpha=0.3, shadow=False, fancybox=True) #, bbox_to_anchor=(1.1, 1))
    plt.savefig('luminosity_isEnt_with_Isochor_cloudy.png', transparent=True)
    plt.show()
    