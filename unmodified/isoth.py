# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:39:12 2022

@author: Alankar
"""

import numpy as np
import sys
sys.path.append('..')
from misc.constants import mp, mH, kpc, K, kB, Xp
from scipy.optimize import root
sys.path.append('..')
sys.path.append('../submodules/AstroPlasma')
from astro_plasma import Ionization
from unmodified.unmodified_profile import UnmodifiedProfile

class IsothermalUnmodified(UnmodifiedProfile):
    
    # Thot is mass wighted average temperature in FM17 (not median Temperature!)
    # P0Tot is in units of kB K cm^-3
    def __init__(self, THot=1.43e6, P0Tot=4580, alpha=1.9, sigmaTurb=60, ZrCGM=0.3, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, redshift=0., ionization='PIE'):
        super().__init__(sigmaTurb=sigmaTurb, ZrCGM=ZrCGM, \
                     M200=M200, MBH=MBH, Mblg=Mblg, \
                     rd=rd, r0=r0, C=C, redshift=redshift, ionization=ionization)
        self.THot = THot*K
        self.P0Tot = P0Tot*kB/(self.__class__.UNIT_DENSITY*self.__class__.UNIT_VELOCITY**2)
        # self.sigmaTh = 141*(km/s)/self.__class__.UNIT_VELOCITY # FSM17 used this adhoc
        # mu here is assumed constant to have sigmaTh a radius independent quantity, hence unmodified profiles are analytic
        mu = 0.61 
        self.sigmaTh = np.sqrt(kB*self.THot/(mu*mp))/self.__class__.UNIT_VELOCITY 
        self.alpha = alpha
        
    def ProfileGen(self, radius_): # Takes in r in kpc, returns Halo density and pressure  in CGS
        super().ProfileGen(radius_)
        if (self._alreadry_acomplished):
            return (self.rho, self.prsTh, self.prsnTh, self.prsTurb, self.prsTot, self.nH, self.mu)
        
        radius = self.radius
        prsTot   = self.P0Tot*np.exp(\
                  ((self.Halo.Phi(radius_)-self.Halo.Phi(self.Halo.r0*self.Halo.UNIT_LENGTH/kpc))\
                  /self.__class__.UNIT_VELOCITY**2)\
                  /(self.alpha*self.sigmaTh**2+self.sigmaTurb**2))\
                   *(self.__class__.UNIT_DENSITY*self.__class__.UNIT_VELOCITY**2) #CGS
        prsTh   = prsTot / (self.alpha+(self.sigmaTurb/self.sigmaTh)**2)
        prsnTh  = (self.alpha-1)*prsTh
        prsTurb = prsTot - (prsTh+prsnTh) 
        self.metallicity  = self.Z0/np.sqrt(1+((radius*kpc/self.UNIT_LENGTH)/self.rZ)**2)
        
        ndens = prsTh/(kB*self.THot)
        nH_guess    = ndens*Xp(self.metallicity)*(mp/mH)*0.61 # guess nH with a guess mu
        nH = np.zeros_like(nH_guess)
        mu = Ionization().interpolate_mu
        for i in range(radius_.shape[0]):
            transcendental = lambda LognH: LognH - \
                np.log10(ndens[i]) - np.log10(Xp(self.metallicity[i])*(mp/mH)*mu(10.**LognH, self.THot , self.metallicity[i], self.redshift, self.ionization) )
            nH[i] = 10.**np.array([root( transcendental, np.log10(nH_guess[i]) ).x[0] ])
                                   
        mu = np.array([ mu(nH[i], self.THot , self.metallicity[i], self.redshift, self.ionization) for i in range(radius_.shape[0])])
        rho = ndens*mu*mp # Gas density only includes thermal component
        self.rho = rho
        self.prsTh = prsTh
        self.prsnTh = prsnTh
        self.prsTurb = prsTurb
        self.prsTot = prsTot
        self.nH = nH
        self.mu = mu
        self.ndens = ndens
        
        return (rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu)
        
