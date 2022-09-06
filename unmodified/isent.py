# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:18:30 2022

@author: Alankar
"""

import sys
sys.path.append('..')
from misc.constants import *
from misc.HaloModel import HaloModel
from scipy.optimize import root

class IsentropicUnmodified:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, nHrCGM=1.1e-5, TthrCGM=2.4e5, sigmaTurb=60, ZrCGM=0.3, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12):
        self.Halo = HaloModel(M200, MBH, Mblg, rd, r0, C)
        
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/IsentropicUnmodified.UNIT_LENGTH
        self.Z0   = 1.0
        self.fZ   = 0.012
        self.alpharCGM = 2.1
        self.gammaTh   = 5/3
        self.gammaNTh  = 4/3
        self.sigmaTurb = (sigmaTurb*km/s)/IsentropicUnmodified.UNIT_VELOCITY
        
        self.ZrCGM = ZrCGM
        self.rZ       = self.rCGM/np.sqrt((self.Z0/self.ZrCGM)**2-1)
        
        self.nHrCGM   = nHrCGM*(cm**(-3))/(IsentropicUnmodified.UNIT_LENGTH**(-3))
        self.nrCGM    = self.nHrCGM*(muHp/mu)
        self.rhorCGM  = (mu*mp/IsentropicUnmodified.UNIT_MASS)*self.nrCGM
        self.TthrCGM  = TthrCGM*K #3.4e5*K*(mu/0.59)*(M200*UNIT_MASS/(1e12*MSun))*(rCGM*UNIT_LENGTH/(300*kpc))**(-1)
        self.KTh      = (kB*self.TthrCGM/(IsentropicUnmodified.UNIT_MASS*IsentropicUnmodified.UNIT_VELOCITY**2))/\
                  ( ((mu*mp/IsentropicUnmodified.UNIT_MASS)**self.gammaTh)*(self.nrCGM**(self.gammaTh-1)) )
        self.KNTh     = ((kB*self.TthrCGM/(IsentropicUnmodified.UNIT_MASS*IsentropicUnmodified.UNIT_VELOCITY**2))/\
                         ( ((mu*mp/IsentropicUnmodified.UNIT_MASS)**self.gammaNTh)*\
                                       (self.nrCGM**(self.gammaNTh-1)) ))*(self.alpharCGM-1)
        self.Db       = self.sigmaTurb**2*np.log(self.rhorCGM) + \
            self.KTh*((self.gammaTh)/(self.gammaTh-1))*(self.rhorCGM**(self.gammaTh-1)) + \
                self.KNTh*((self.gammaNTh)/(self.gammaNTh-1))*(self.rhorCGM**(self.gammaNTh-1))
                
    def _transcendental(self, log10x, rad): 
        rho = 10**log10x
        return self.sigmaTurb**2*np.log(rho) + \
            self.KTh*((self.gammaTh)/(self.gammaTh-1))*(rho**(self.gammaTh-1)) +\
                self.KNTh*((self.gammaNTh)/(self.gammaNTh-1))*(rho**(self.gammaNTh-1)) -\
                    self.Db + \
                    ((-self.Halo.Phi(rad*IsentropicUnmodified.UNIT_LENGTH/kpc)+\
                     self.Halo.Phi(self.rCGM*IsentropicUnmodified.UNIT_LENGTH/kpc))/IsentropicUnmodified.UNIT_VELOCITY**2)
                    
    def ProfileGen(self, radius_):  #takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
           
        radius = np.copy(radius_)*kpc/IsentropicUnmodified.UNIT_LENGTH
        nH_guess =  np.logspace(-3*(8.5/radius_[0]),-5*(220/radius_[-1]), len(radius)) #CGS
        rho_guess = np.log10(nH_guess*muHp*mp/IsentropicUnmodified.UNIT_DENSITY)
        #rho_guess = rhorCGM - 0.93*np.log10(radius/rCGM)
        #rho_guess = np.log10(rhorCGM*(radius/rCGM)**(-0.93))
        rho    = 10**(np.array([root(self._transcendental, rho_guess[indx] , 
                                args=(rad,)).x[0] for indx, rad in enumerate(radius)]))
        
        prsTh   = self.KTh*(rho**self.gammaTh)*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        prsnTh  = self.KNTh*(rho**self.gammaNTh)*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        prsTurb = rho*self.sigmaTurb**2*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        prsTot  = prsTh + prsnTh + prsTurb
        
        if (len(radius_)==1): 
            return (rho[0]*IsentropicUnmodified.UNIT_DENSITY, prsTh[0], prsnTh[0], prsTurb[0], prsTot[0])
        else:
            return (rho*IsentropicUnmodified.UNIT_DENSITY, prsTh, prsnTh, prsTurb, prsTot)