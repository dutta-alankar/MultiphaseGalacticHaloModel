# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:18:30 2022

@author: Alankar
"""

import numpy as np
import sys
sys.path.append('..')
from misc.constants import mp, mH, kpc, K, cm, kB, Xp
from scipy.optimize import root
from dataclasses import dataclass
from typing import Union, Tuple
sys.path.append('..')
sys.path.append('../submodules/AstroPlasma')
from astro_plasma import Ionization
from unmodified.unmodified_profile import UnmodifiedProfile

@dataclass
class IsentropicUnmodified(UnmodifiedProfile):
    
    nHrCGM: float = 1.1e-5
    TthrCGM: float = 2.4e5
    unmod_type: str = "isent"
    
    def __post_init__(self: "IsentropicUnmodified") -> None:
        super().__post_init__()
        self.fZ   = 0.012
        self.alpharCGM = 2.1
        self.gammaTh   = 5/3
        self.gammaNTh  = 4/3
        self.TthrCGM  = self.TthrCGM*K #3.4e5*K*(mu/0.59)*(M200*UNIT_MASS/(1e12*MSun))*(rCGM*UNIT_LENGTH/(300*kpc))**(-1)
        self.nHrCGM   = self.nHrCGM*(cm**(-3))/(self.UNIT_LENGTH**(-3))
        
        mu = Ionization.interpolate_mu
        self.metallicityrCGM = self.Z0/np.sqrt(1+(self.rCGM/self.rZ)**2)
        self.murCGM = mu(self.nHrCGM*(self.UNIT_LENGTH**(-3)), \
                         self.TthrCGM, self.metallicityrCGM, self.redshift, self.ionization)          
        self.nrCGM    = self.nHrCGM*(mH/mp)/(Xp(self.metallicityrCGM)*self.murCGM) # code
        self.rhorCGM  = self.nHrCGM*mH/Xp(self.metallicityrCGM)*(self.UNIT_LENGTH**(-3))/self.UNIT_DENSITY
        self.KTh      = (kB*self.TthrCGM/(self.UNIT_MASS*self.UNIT_VELOCITY**2))/\
                        ( ((self.murCGM*mp/self.UNIT_MASS)**self.gammaTh)*(self.nrCGM**(self.gammaTh-1)) ) # code
        self.KNTh     = ((kB*self.TthrCGM/(self.UNIT_MASS*self.UNIT_VELOCITY**2))/\
                        ( ((self.murCGM*mp/self.UNIT_MASS)**self.gammaNTh)*\
                        (self.nrCGM**(self.gammaNTh-1)) ))*(self.alpharCGM-1) # code
        self.Db       = self.sigmaTurb**2*np.log(self.rhorCGM) + \
                        self.KTh*((self.gammaTh)/(self.gammaTh-1))*(self.rhorCGM**(self.gammaTh-1)) + \
                        self.KNTh*((self.gammaNTh)/(self.gammaNTh-1))*(self.rhorCGM**(self.gammaNTh-1)) # code
                
                    
    def ProfileGen(self: "IsentropicUnmodified", radius_: Union[float, list, np.ndarray]) -> Tuple:  # Takes in r in kpc, returns Halo density and pressure  in CGS       
        super().ProfileGen(radius_)
        if (self._alreadry_acomplished):
            return (self.rho, self.prsTh, self.prsnTh, self.prsTurb, self.prsTot, self.nH, self.mu)
        
        radius = self.radius
        mu = Ionization.interpolate_mu
        
        def _transcendental(log10rho: float, rad: float) -> float: 
            rho = 10**log10rho
            return self.sigmaTurb**2*np.log(rho) + \
                self.KTh*((self.gammaTh)/(self.gammaTh-1))*(rho**(self.gammaTh-1)) +\
                    self.KNTh*((self.gammaNTh)/(self.gammaNTh-1))*(rho**(self.gammaNTh-1)) -\
                        self.Db + \
                        ((-self.Halo.Phi(rad*self.UNIT_LENGTH/kpc)+\
                         self.Halo.Phi(self.rCGM*self.UNIT_LENGTH/kpc))/self.UNIT_VELOCITY**2)
        
        nH_guess =  np.logspace(-3*(8.5/radius_[0]),-5*(220/radius_[-1]), len(radius)) # CGS
        rho_guess = np.log10((nH_guess*mH/Xp(self.metallicity))/self.UNIT_DENSITY)
        
        rho    = 10.**(np.array([root(_transcendental, rho_guess[indx], 
                                args=(rad,)).x[0] for indx, rad in enumerate(radius)])) # code
        
        self.prsTh   = self.KTh*(rho**self.gammaTh)*self.UNIT_DENSITY*self.UNIT_VELOCITY**2
        self.prsnTh  = self.KNTh*(rho**self.gammaNTh)*self.UNIT_DENSITY*self.UNIT_VELOCITY**2
        self.prsTurb = rho*self.sigmaTurb**2*self.UNIT_DENSITY*self.UNIT_VELOCITY**2
        self.prsTot  = self.prsTh + self.prsnTh + self.prsTurb
        
        self.rho = rho*self.UNIT_DENSITY
        self.nH = self.rho*Xp(self.metallicity)/mH
        
        self.Temperature = np.zeros_like(radius)
        
        for i in range(radius.shape[0]):
            transcendental = lambda LogTemp: np.log10(((10.**LogTemp)/mu(self.nH[i], 10.**LogTemp, self.metallicity[i], self.redshift, self.ionization))) \
                                        - np.log10(((self.prsTh[i]/kB)/(self.rho[i]/mp)))
            logT_guess = np.log10(((self.prsTh[i]/kB)/(self.rho[i]/mp))*0.61)
            self.Temperature[i] = 10.**(root(transcendental, logT_guess, method='lm', tol=1e-8).x[0])
        
        self.mu = self.Temperature/((self.prsTh/kB)/(self.rho/mp))
        self.ndens = self.rho/(self.mu*mp)
            
        return (self.rho, self.prsTh, self.prsnTh, self.prsTurb, self.prsTot, self.nH, self.mu)
