#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 10:48:42 2023

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import matplotlib.gridspec as gridspec
import observable.maps as maps
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from observable.EmissionMeasure import EmissionMeasure as EM
from figures.diskEM import EmissionMeasureDisk as EMDisk
import os

do_isothermal, do_isentropic = True, True
showProgress = False

b = np.linspace(-90, 90, 180)
l = np.linspace(0., 360, 360)

l, b = np.meshgrid(l, b) 

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
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionDisk = EMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./emissionDisk.npy', emissionDisk)
    
    emission_PIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isoth/emission_PIE_lb-ic.npy', emission_PIE)
    
    
    # CIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionDisk = EMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    emission_CIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isoth/emission_CIE_lb-ic.npy', emission_CIE)
    
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
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionDisk = EMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    emission_PIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isent/emission_PIE_lb-ic.npy', emission_PIE)
    
    
    # CIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionDisk = EMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    emission_CIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isent/emission_CIE_lb-ic.npy', emission_CIE)