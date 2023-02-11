#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 13:33:06 2022

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
from observable.EmissionSpectrum import LuminositySpectrumGen
import os

do_isothermal, do_isentropic = True, True
# soft X-ray eFEDs
Emin  = 0.30
Emax1 = 0.60
Emax2 = 2.0

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
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    spectrum_PIE = LuminositySpectrumGen(mod_isochor)
    
    np.save("./isoth/emm-spec-ic_PIE.npy", spectrum_PIE)
        
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax1)
    SB_PIE = np.trapz(spectrum_PIE[select,1], spectrum_PIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB PIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax1, SB_PIE))
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax2)
    SB_PIE = np.trapz(spectrum_PIE[select,1], spectrum_PIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB PIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax2, SB_PIE))
    
    # CIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    spectrum_CIE = LuminositySpectrumGen(mod_isochor)
    
    np.save("./isoth/emm-spec-ic_CIE.npy", spectrum_CIE)
        
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax1)
    SB_CIE = np.trapz(spectrum_CIE[select,1], spectrum_CIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB CIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax1, SB_CIE))
    select = np.logical_and(spectrum_CIE[:,0]>=Emin, spectrum_CIE[:,0]<=Emax2)
    SB_CIE = np.trapz(spectrum_CIE[select,1], spectrum_CIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB CIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax2, SB_CIE))
    
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
    cutoff = 8.0
    
    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    spectrum_PIE = LuminositySpectrumGen(mod_isochor)
    
    np.save("./isent/emm-spec-ic_PIE.npy", spectrum_PIE)
        
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax1)
    SB_PIE = np.trapz(spectrum_PIE[select,1], spectrum_PIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB PIE isentrp (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax1, SB_PIE))
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax2)
    SB_PIE = np.trapz(spectrum_PIE[select,1], spectrum_PIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB PIE isentrp (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax2, SB_PIE))
    
    # CIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    spectrum_CIE = LuminositySpectrumGen(mod_isochor)
    
    np.save("./isent/emm-spec-ic_CIE.npy", spectrum_CIE)
    
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    select = np.logical_and(spectrum_PIE[:,0]>=Emin, spectrum_PIE[:,0]<=Emax1)
    SB_CIE = np.trapz(spectrum_CIE[select,1], spectrum_CIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB CIE isentrp (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax1, SB_CIE))
    select = np.logical_and(spectrum_CIE[:,0]>=Emin, spectrum_CIE[:,0]<=Emax2)
    SB_CIE = np.trapz(spectrum_CIE[select,1], spectrum_CIE[select,0])/(4*(np.pi*rCGM)**2)
    print("SB CIE isentrp (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax2, SB_CIE))
    
