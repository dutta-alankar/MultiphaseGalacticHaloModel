#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:34:18 2022

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
from observable.DispersionMeasure import DispersionMeasure as DM
from figures.diskDM import DispersionMeasureDisk as DMDisk
import os

do_isothermal, do_isentropic = False, True
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
    
    dispersionDisk = DMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./dispersionDisk.npy', dispersionDisk)
    
    dispersion_PIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isoth/dispersion_PIE_lb-ic.npy', dispersion_PIE)
    
    
    # CIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    dispersionDisk = DMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    dispersion_CIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isoth/dispersion_CIE_lb-ic.npy', dispersion_CIE)
    
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
    
    dispersionDisk = DMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    dispersion_PIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isent/dispersion_PIE_lb-ic.npy', dispersion_PIE)
    
    
    # CIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    dispersionDisk = DMDisk(mod_isochor).generate(l, b, showProgress=showProgress)
    
    dispersion_CIE = DM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('./isent/dispersion_CIE_lb-ic.npy', dispersion_CIE)
    