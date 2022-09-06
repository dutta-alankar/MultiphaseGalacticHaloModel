# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:39:12 2022

@author: Alankar
"""

import sys
sys.path.append('..')
from misc.constants import *
from misc.HaloModel import HaloModel

class IsothermalUnmodified:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    #Thot is mass wighted average temperature in FM17 (not median Temperature!)
    #P0Tot is in units of kB K cm^-3
    def __init__(self, THot=1.43e6, P0Tot=4580, alpha=1.9, sigmaTurb=60, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12):
        self.Halo = HaloModel(M200, MBH, Mblg, rd, r0, C)
        self.THot = THot*K
        self.P0Tot = P0Tot*kB/(IsothermalUnmodified.UNIT_DENSITY*IsothermalUnmodified.UNIT_VELOCITY**2)
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/IsothermalUnmodified.UNIT_LENGTH
        #self.sigmaTh = 141*(km/s)/IsothermalUnmodified.UNIT_VELOCITY #FSM17 used this adhoc
        self.sigmaTh = np.sqrt(kB*self.THot/(mu*mp))/IsothermalUnmodified.UNIT_VELOCITY #mu=1.0
        self.alpha = alpha
        self.sigmaTurb = sigmaTurb*(km/s)/IsothermalUnmodified.UNIT_VELOCITY
        
    def ProfileGen(self, radius_): #takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
            
        radius = np.copy(radius_)*kpc/IsothermalUnmodified.UNIT_LENGTH
        prsTot   = self.P0Tot*np.exp(\
                  ((self.Halo.Phi(radius_)-self.Halo.Phi(self.Halo.r0*self.Halo.UNIT_LENGTH/kpc))\
                  /IsothermalUnmodified.UNIT_VELOCITY**2)\
                  /(self.alpha*self.sigmaTh**2+self.sigmaTurb**2))\
                   *(IsothermalUnmodified.UNIT_DENSITY*IsothermalUnmodified.UNIT_VELOCITY**2) #CGS
        prsTh   = prsTot / (self.alpha+(self.sigmaTurb/self.sigmaTh)**2)
        prsnTh  = (self.alpha-1)*prsTh
        prsTurb = prsTot - (prsTh+prsnTh) 
        
        rho = (prsTh/self.THot)*((mu*mp)/kB) # gas density only includes thermal component
        #rho = (prsTot/self.THot)*((mu*mp)/kB)
        
        if (len(radius_)==1): 
            return (rho[0], prsTh[0], prsnTh[0], prsTurb[0], prsTot[0])
        else:
            return (rho, prsTh, prsnTh, prsTurb, prsTot)
        