# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:39:12 2022

@author: Alankar
"""

import sys
sys.path.append('..')
from misc.constants import *
from misc.HaloModel import HaloModel
from scipy.optimize import root
from misc.ionization import interpolate_ionization

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
    def __init__(self, THot=1.43e6, P0Tot=4580, alpha=1.9, sigmaTurb=60, ZrCGM=0.3, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, redshift=0., ionization='PIE'):
        self.Halo = HaloModel(M200, MBH, Mblg, rd, r0, C)
        self.THot = THot*K
        self.P0Tot = P0Tot*kB/(IsothermalUnmodified.UNIT_DENSITY*IsothermalUnmodified.UNIT_VELOCITY**2)
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/IsothermalUnmodified.UNIT_LENGTH
        self.Z0   = 1.0
        self.ZrCGM = ZrCGM
        self.rZ    = self.rCGM/np.sqrt((self.Z0/self.ZrCGM)**2-1)
        #self.sigmaTh = 141*(km/s)/IsothermalUnmodified.UNIT_VELOCITY #FSM17 used this adhoc
        self.sigmaTh = np.sqrt(kB*self.THot/(mu*mp))/IsothermalUnmodified.UNIT_VELOCITY #mu=1.0
        self.alpha = alpha
        self.sigmaTurb = sigmaTurb*(km/s)/IsothermalUnmodified.UNIT_VELOCITY
        self.redshift = redshift
        self.ionization = ionization
        
    def ProfileGen(self, radius_): #takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
            
        radius = np.copy(radius_)*kpc/IsothermalUnmodified.UNIT_LENGTH
        prsTot   = self.P0Tot*np.exp(\
                  ((self.Halo.Phi(radius_)-self.Halo.Phi(self.Halo.r0*self.Halo.UNIT_LENGTH/kpc))\
                  /IsothermalUnmodified.UNIT_VELOCITY**2)\
                  /(self.alpha*self.sigmaTh**2+self.sigmaTurb**2))\
                   *(IsothermalUnmodified.UNIT_DENSITY*IsothermalUnmodified.UNIT_VELOCITY**2) #CGS
        prsTh   = prsTot / (self.alpha+(self.sigmaTurb/self.sigmaTh)**2)
        prsnTh  = (self.alpha-1)*prsTh
        prsTurb = prsTot - (prsTh+prsnTh) 
        self.metallicity  = self.Z0/np.sqrt(1+((radius*kpc/self.UNIT_LENGTH)/self.rZ)**2)
        
        ndens = prsTh/(kB*self.THot)
        nH_guess    = ndens*Xp(self.metallicity)*(mp/mH)*0.61 #guess nH
        nH = np.zeros_like(nH_guess)
        mu = interpolate_ionization().mu
        #print("calculating unmodified nH")
        for i in range(radius_.shape[0]):
            # transcendental = lambda LognH: 10.**LognH - \
            #     ndens[i]*Xp(self.metallicity[i])*(mp/mH)*mu(10.**LognH, self.THot , self.metallicity[i], self.redshift, self.ionization)
            transcendental = lambda LognH: LognH - \
                np.log10(ndens[i]) - np.log10(Xp(self.metallicity[i])*(mp/mH)*mu(10.**LognH, self.THot , self.metallicity[i], self.redshift, self.ionization) )
            nH[i] = 10.**np.array([root( transcendental, np.log10(nH_guess[i]) ).x[0] ])
                                   
        mu = np.array([ mu(nH[i], self.THot , self.metallicity[i], self.redshift, self.ionization) for i in range(radius_.shape[0])])
        rho = ndens*mu*mp # gas density only includes thermal component
        #rho = (prsTot/self.THot)*((mu*mp)/kB)
        #print("finished calculating unmodified nH")
        self.rho = rho
        self.prsTh = prsTh
        self.prsnTh = prsnTh
        self.prsTurb = prsTurb
        self.prsTot = prsTot
        self.nH = nH
        self.mu = mu
        self.ndens = ndens
        
        # if (len(radius_)==1): 
        #     return (rho[0], prsTh[0], prsnTh[0], prsTurb[0], prsTot[0], nH[0], mu[0])
        # else:
        return (rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu)
        
