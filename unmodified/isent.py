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
from scipy.interpolate import interp1d
from misc.ionization import interpolate_ionization

class IsentropicUnmodified:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, nHrCGM=1.1e-5, TthrCGM=2.4e5, sigmaTurb=60, ZrCGM=0.3, \
                 M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, redshift=0., ionization='PIE'):
        self.Halo = HaloModel(M200, MBH, Mblg, rd, r0, C)
        
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/IsentropicUnmodified.UNIT_LENGTH
        self.Z0   = 1.0 #ISM
        self.fZ   = 0.012
        self.alpharCGM = 2.1
        self.gammaTh   = 5/3
        self.gammaNTh  = 4/3
        self.sigmaTurb = (sigmaTurb*km/s)/IsentropicUnmodified.UNIT_VELOCITY
        self.TthrCGM  = TthrCGM*K #3.4e5*K*(mu/0.59)*(M200*UNIT_MASS/(1e12*MSun))*(rCGM*UNIT_LENGTH/(300*kpc))**(-1)
        self.nHrCGM   = nHrCGM*(cm**(-3))/(IsentropicUnmodified.UNIT_LENGTH**(-3))
        self.ZrCGM = ZrCGM
        self.rZ       = self.rCGM/np.sqrt((self.Z0/self.ZrCGM)**2-1)
        self.redshift = redshift
        self.ionization = ionization
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        
        mu = interpolate_ionization().mu
        self.metallicityrCGM = self.Z0/np.sqrt(1+(self.rCGM/self.rZ)**2)
        
        self.murCGM = mu(self.nHrCGM*(IsentropicUnmodified.UNIT_LENGTH**(-3)), \
                         self.TthrCGM, self.metallicityrCGM, self.redshift, self.ionization) 
              
        self.nrCGM    = self.nHrCGM*(mH/mp)/(Xp(self.metallicityrCGM)*self.murCGM) #code
        self.rhorCGM  = self.nHrCGM*mH/Xp(self.metallicityrCGM)*(IsentropicUnmodified.UNIT_LENGTH**(-3))/IsentropicUnmodified.UNIT_DENSITY
        self.KTh      = (kB*self.TthrCGM/(IsentropicUnmodified.UNIT_MASS*IsentropicUnmodified.UNIT_VELOCITY**2))/\
                        ( ((self.murCGM*mp/IsentropicUnmodified.UNIT_MASS)**self.gammaTh)*(self.nrCGM**(self.gammaTh-1)) ) #code
        self.KNTh     = ((kB*self.TthrCGM/(IsentropicUnmodified.UNIT_MASS*IsentropicUnmodified.UNIT_VELOCITY**2))/\
                        ( ((self.murCGM*mp/IsentropicUnmodified.UNIT_MASS)**self.gammaNTh)*\
                        (self.nrCGM**(self.gammaNTh-1)) ))*(self.alpharCGM-1) #code
        self.Db       = self.sigmaTurb**2*np.log(self.rhorCGM) + \
                        self.KTh*((self.gammaTh)/(self.gammaTh-1))*(self.rhorCGM**(self.gammaTh-1)) + \
                        self.KNTh*((self.gammaNTh)/(self.gammaNTh-1))*(self.rhorCGM**(self.gammaNTh-1)) #code
                
    def _transcendental(self, log10rho, rad): 
        rho = 10**log10rho
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
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        
        mu = interpolate_ionization().mu
        
        self.metallicity  = self.Z0/np.sqrt(1+((radius*kpc/self.UNIT_LENGTH)/self.rZ)**2)
        
        nH_guess =  np.logspace(-3*(8.5/radius_[0]),-5*(220/radius_[-1]), len(radius)) #CGS
        rho_guess = np.log10((nH_guess*mH/Xp(self.metallicity))/IsentropicUnmodified.UNIT_DENSITY)
        
        rho    = 10.**(np.array([root(self._transcendental, rho_guess[indx], 
                                args=(rad,)).x[0] for indx, rad in enumerate(radius)])) #code
        
        self.prsTh   = self.KTh*(rho**self.gammaTh)*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        self.prsnTh  = self.KNTh*(rho**self.gammaNTh)*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        self.prsTurb = rho*self.sigmaTurb**2*IsentropicUnmodified.UNIT_DENSITY*IsentropicUnmodified.UNIT_VELOCITY**2
        self.prsTot  = self.prsTh + self.prsnTh + self.prsTurb
        
        self.rho = rho*IsentropicUnmodified.UNIT_DENSITY
        self.nH = self.rho*Xp(self.metallicity)/mH
        
        self.Temperature = np.zeros_like(radius)
        
        for i in range(radius.shape[0]):
            transcendental = lambda LogTemp: ((10.**LogTemp)/mu(self.nH[i], 10.**LogTemp, self.metallicity[i], self.redshift, self.ionization)) \
                                        - ((self.prsTh[i]/kB)/(self.rho[i]/mp))
            logT_guess = np.log10(((self.prsTh[i]/kB)/(self.rho[i]/mp))*0.61)
            self.Temperature[i] = 10.**(root(transcendental, logT_guess).x[0])
        
        self.mu = self.Temperature/((self.prsTh/kB)/(self.rho/mp))
        self.ndens = self.rho/(self.mu*mp)
            
        # if (len(radius_)==1): 
        #     return (self.rho[0], self.prsTh[0], self.prsnTh[0], self.prsTurb[0], self.prsTot[0])
        # else:
        return (self.rho, self.prsTh, self.prsnTh, self.prsTurb, self.prsTot, self.nH, self.mu)
