#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 11:37:48 2023

@author: alankar
"""

import numpy as np
from scipy.optimize import root
import sys
sys.path.append('..')
from misc.ionization import interpolate_ionization

ionization = interpolate_ionization()
num_dens = ionization.num_dens 

nH = 1.2e-2 # Kaaret       
ne = 3.8e-3 # Yamasaki

metallicity = 1.0
redshift = 0.003
mode = 'PIE'

ne_converge = lambda logTDisk: np.log10(num_dens(nH, 10.**logTDisk, metallicity, redshift, mode, 'electron')/ne)

TDisk = 10.**root( ne_converge, np.log10(2e4) ).x[0]

print('TDisk = %.2e'%TDisk)