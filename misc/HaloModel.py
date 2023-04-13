# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:04:53 2022

@author: Alankar
"""
import sys
sys.path.append('..')
from misc.constants import kpc, mp, km, s, K, MSun, dcrit0, G
from dataclasses import dataclass
import numpy as np
from scipy import integrate

@dataclass
class HaloModel:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    M200: float = 1e12
    MBH: float = 2.6e6
    Mblg: float = 6e10
    rd: float = 3.0
    r0: float = 8.5
    C: float = 12
    _withDisk: bool = True
    
    def __post_init__(self, ):
        self.M200 = self.M200*MSun/self.UNIT_MASS
        self.MBH = self.MBH*MSun/self.UNIT_MASS
        self.Mb = self.Mblg*MSun/self.UNIT_MASS
        self.rd = self.rd*kpc/self.UNIT_LENGTH
        self.r0 = self.r0*kpc/self.UNIT_LENGTH
        self.C  = self.C
        # self.r200 = 258*kpc/self.UNIT_LENGTH  # Test with Faerman but not self-consistent
        self.r200 = (3*self.M200/(800*np.pi*(dcrit0/self.UNIT_DENSITY)))**(1/3.)
        self.rs = self.r200/self.C
     
    def Mass(self, r): # takes in r in kpc, returns Halo Mass in CGS
        r_ = r*kpc/self.UNIT_LENGTH
        f     = lambda x: np.log(1+x)-x/(1+x)
        MHalo = lambda rp :  self.M200*f(rp/self.rs)/f(self.C) # NFW
        MDisk = lambda rp : (0.025*(1-np.exp(-2.64*((rp/(kpc/self.UNIT_LENGTH)))**1.15)) + \
                                0.142*(1-(1+(rp/(kpc/self.UNIT_LENGTH))**1.5)*\
                                np.exp(-(rp/((kpc/self.UNIT_LENGTH)))**1.5)) + \
                                0.833*(1-(1+rp/self.rd)*np.exp(-rp/self.rd)))*self.Mb
        return (self.MBH + MHalo(r_) + (MDisk(r_) if self._withDisk else 0.) )*self.UNIT_MASS
    
    def Phi(self, r): #takes in r in kpc, returns Halo Potential in CGS
        r_ = r*kpc/self.UNIT_LENGTH
        _phi = lambda rpp : -(G*self.UNIT_LENGTH**2*self.UNIT_DENSITY/self.UNIT_VELOCITY**2)*\
            (integrate.quad(lambda rp: (self.Mass(rp)/self.UNIT_MASS)/rp**2, self.r0, rpp)[0])
        if isinstance(r_, list) : 
            R = np.array(r_)
        elif isinstance(r_, np.ndarray):
            R = np.copy(r_)
        else: 
            return _phi(r_)*self.UNIT_VELOCITY**2   
        return np.array([_phi(rad) for rad in R])*self.UNIT_VELOCITY**2
