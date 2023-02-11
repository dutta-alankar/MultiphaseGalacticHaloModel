#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:59:53 2022

@author: alankar
"""
import sys
sys.path.append('..')
import numpy as np
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution

do_isothermal, do_isentropic = True, True
Emin = 0.3
Emax = [0.6, 2.0]

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    print('Isothermal:')
    isoth_ic_CIE = np.loadtxt('emm-spec-isth-ic_CIE.txt')
    isoth_ic_PIE = np.loadtxt('emm-spec-isth-ic_PIE.txt')
    
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
    
    energy = isoth_ic_PIE[:,0]
    spectrum = isoth_ic_PIE[:,1]
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    
    select = np.logical_and(energy>=Emin, energy<=Emax[0])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB PIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[0], SB))
    
    select = np.logical_and(energy>=Emin, energy<=Emax[1])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB PIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[1], SB))
    
    # CIE
    print()
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    energy = isoth_ic_CIE[:,0]
    spectrum = isoth_ic_CIE[:,1] 
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    
    select = np.logical_and(energy>=Emin, energy<=Emax[0])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB CIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[0], SB))
    
    select = np.logical_and(energy>=Emin, energy<=Emax[1])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB CIE isotherm (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[1], SB))
    
# ____________________________________________________________
# _________________ Isentropic profile _______________________
print()
if(do_isentropic):
    print('Isentropic:')
    isent_ic_CIE = np.loadtxt('emm-spec-isent-ic_CIE.txt')
    isent_ic_PIE = np.loadtxt('emm-spec-isent-ic_PIE.txt')  
    
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
    
    energy = isent_ic_PIE[:,0]
    spectrum = isent_ic_PIE[:,1]
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    
    select = np.logical_and(energy>=Emin, energy<=Emax[0])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB PIE isentrop (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[0], SB))
    
    select = np.logical_and(energy>=Emin, energy<=Emax[1])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB PIE isentrop (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[1], SB))
    
    # CIE
    print()
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
     
    energy = isent_ic_CIE[:,0]
    spectrum = isent_ic_CIE[:,1]
    rCGM = mod_isochor.unmodified.rCGM*mod_isochor.unmodified.UNIT_LENGTH
    
    select = np.logical_and(energy>=Emin, energy<=Emax[0])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB CIE isentrop (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[0], SB))
    
    select = np.logical_and(energy>=Emin, energy<=Emax[1])
    # print(np.sum(select))
    SB = np.trapz(spectrum[select], energy[select])/(4*(np.pi*rCGM)**2)
    print("SB CIE isentrop (%.1f-%.1f keV): %.2e erg cm^-2 s^-1 deg^-2"%(Emin, Emax[1], SB))
