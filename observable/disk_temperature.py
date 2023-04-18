# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:39:22 2023

@author: alankar

This code isn't working as expected as of now!
"""

import sys

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")

import numpy as np
from scipy.optimize import root
from astro_plasma import Ionization

num_dens = Ionization.interpolate_num_dens

nH = 1.2e-3 # Kaaret       
ne = 3.8e-3 # Yamasaki

ne_target = nH*(1/0.71+(1-0/16)*1.4e-2)

metallicity = 1.0
redshift = 1.0
mode = 'PIE'

def ne_converge(logTDisk): 
    ne = num_dens(nH, 10.**logTDisk, metallicity, redshift, mode, 'electron') 
    print(ne, 10.**logTDisk, nH/ne)
    return ne - ne_target

TDisk = 10.**root( ne_converge, np.log10(3e6) ).x[0]

print('TDisk = %.2e'%TDisk)