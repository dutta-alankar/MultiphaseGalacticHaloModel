#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:21:15 2023

@author: alankar
"""

import numpy as np
import sys
sys.path.append('..')
from misc.constants import mp, kpc, km, s, K
from misc.HaloModel import HaloModel

class UnmodifiedProfile:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, sigmaTurb=60, ZrCGM=0.3, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, \
                 rd=3.0, r0=8.5, C=12, redshift=0., ionization='PIE'):
        self.Halo = HaloModel(M200, MBH, Mblg, rd, r0, C)
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/self.__class__.UNIT_LENGTH
        self.Z0   = 1.0
        self.ZrCGM = ZrCGM
        self.rZ    = self.rCGM/np.sqrt((self.Z0/self.ZrCGM)**2-1)
        self.sigmaTurb = sigmaTurb*(km/s)/self.__class__.UNIT_VELOCITY
        self.redshift = redshift
        self.ionization = ionization
        
    def ProfileGen(self, radius_):  #takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        
        # Check if already generated then don't redo any work
        self._alreadry_acomplished = False
        if hasattr(self, "radius"):
            if (np.sum(np.copy(radius_)*kpc/self.__class__.UNIT_LENGTH == self.radius) == self.radius.shape[0]):
                self._alreadry_acomplished = True
                
        self.radius = np.copy(radius_)*kpc/self.__class__.UNIT_LENGTH
        self.metallicity = self.Z0/np.sqrt(1+((self.radius*kpc/self.__class__.UNIT_LENGTH)/self.rZ)**2)