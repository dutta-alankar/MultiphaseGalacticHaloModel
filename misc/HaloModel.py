# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:04:53 2022

@author: Alankar
"""
import sys
sys.path.append('..')
from misc.constants import *
import numpy as np
from scipy import integrate

class HaloModel:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12):
        self.M200 = M200*MSun/HaloModel.UNIT_MASS
        self.MBH = MBH*MSun/HaloModel.UNIT_MASS
        self.Mb = Mblg*MSun/HaloModel.UNIT_MASS
        self.rd = rd*kpc/HaloModel.UNIT_LENGTH
        self.r0 = r0*kpc/HaloModel.UNIT_LENGTH
        self.C  = C
        #self.r200 = 258*kpc/HaloModel.UNIT_LENGTH  # Test with Faerman but not self-consistent
        self.r200 = (3*self.M200/(800*pi*(dcrit0/HaloModel.UNIT_DENSITY)))**(1/3.)
        self.rs = self.r200/self.C
     
    def Mass(self, r): #takes in r in kpc, returns Halo Mass in CGS
        r_ = r*kpc/HaloModel.UNIT_LENGTH
        f     = lambda x: np.log(1+x)-x/(1+x)
        MHalo = lambda rp :  self.M200*f(rp/self.rs)/f(self.C) #NFW
        MDisk = lambda rp : (0.025*(1-np.exp(-2.64*((rp/(kpc/HaloModel.UNIT_LENGTH)))**1.15)) + \
                                0.142*(1-(1+(rp/(kpc/HaloModel.UNIT_LENGTH))**1.5)*\
                                np.exp(-(rp/((kpc/HaloModel.UNIT_LENGTH)))**1.5)) + \
                                0.833*(1-(1+rp/self.rd)*np.exp(-rp/self.rd)))*self.Mb
        return (self.MBH + MHalo(r_) + MDisk(r_))*HaloModel.UNIT_MASS
    
    def Phi(self, r): #takes in r in kpc, returns Halo Potential in CGS
        r_ = r*kpc/HaloModel.UNIT_LENGTH
        _phi = lambda rpp : -(G*HaloModel.UNIT_LENGTH**2*HaloModel.UNIT_DENSITY/HaloModel.UNIT_VELOCITY**2)*\
            (integrate.quad(lambda rp: (self.Mass(rp)/HaloModel.UNIT_MASS)/rp**2, self.r0, rpp)[0])
        if isinstance(r_, list) : 
            R = np.array(r_)
        elif isinstance(r_, np.ndarray):
            R = np.copy(r_)
        else: 
            return _phi(r_)*HaloModel.UNIT_VELOCITY**2   
        return np.array([_phi(rad) for rad in R])*HaloModel.UNIT_VELOCITY**2