#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 20:56:09 2022

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
from misc.ProgressBar import ProgressBar
from  misc.constants import *
import observable.CoordinateTrans as transform
import observable.maps as maps
from misc.ionization import interpolate_ionization

class EmissionMeasureDisk(maps.MapInit):
    
    def __init__(self, redisProf, mode='PIE'):
        super().__init__(redisProf)
        self.redisProf = redisProf
        self.mode = mode
            
    def generate(self, l, b, showProgress=True):
        l, b = super().prepare(l,b)
        rend = self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
        
        ionization = interpolate_ionization()
        num_dens = ionization.num_dens    
        def nH_prof(R, z):
            n0 = 1.2e-2 #cm^-3
            R0 = 5.4 #kpc
            z0 = 2.8 #kpc
            return n0 * np.exp(-(R/R0 + np.fabs(z)/z0))
        metallicity = 1.0
        redshift = 0.001
        mode = self.mode
        TDisk = 4.25e3 if mode=='PIE' else 1.63e4 # K
        
        ne0 = num_dens(nH_prof(0.,0.), TDisk, metallicity, redshift, mode, 'electron')
        ni0 = num_dens(nH_prof(0.,0.), TDisk, metallicity, redshift, mode, 'ion')
        
        def ne_prof(R, z):
            n0 = 1.2e-2 #cm^-3
            R0 = 5.4 #kpc
            z0 = 2.8 #kpc
            return ne0 * np.exp(-(R/R0 + np.fabs(z)/z0))
        
        def ni_prof(R, z):
            n0 = 1.2e-2 #cm^-3
            R0 = 5.4 #kpc
            z0 = 2.8 #kpc
            return ni0 * np.exp(-(R/R0 + np.fabs(z)/z0))
        
        if isinstance(l, np.ndarray):          
            if showProgress: 
                progBar = None
                self.EM = np.zeros_like(l)
                for i in range(self.EM.shape[0]):
                    for j in range(self.EM.shape[1]):
                        LOSsample = np.logspace(np.log10(1e-3*self.integrateTill[i,j]), 
                                                np.log10(self.integrateTill[i,j]), 100) 
                        radius, phi, theta = transform.toGalC(l[i,j], b[i,j], LOSsample)
                        height = np.abs(radius*np.cos(np.deg2rad(theta)))
                        radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                        
                        ne = ne_prof(radius, height)
                        ni = ni_prof(radius, height)
                        
                        em = np.trapz( np.nan_to_num(ne*ni) , LOSsample)
                        
                        # print('em= ', em)
                        self.EM[i,j] =  em #cm^-6 kpc
                        if (i==0 and j==0): progBar = ProgressBar()
                        progBar.progress(i*self.EM.shape[1]+j+1, 
                                         self.EM.shape[0]*self.EM.shape[1])
                progBar.end()
            else:
                def _calc(tup):
                    l_val, b_val, integrateTill = tup
                    LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100) #np.logspace(-6, np.log10(integrateTill), 10)
                    radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
                    height = np.abs(radius*np.cos(np.deg2rad(theta)))
                    radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                    
                    ne = ne_prof(radius, height)
                    ni = ni_prof(radius, height)
                    
                    em = np.trapz( np.nan_to_num(ne*ni), LOSsample )
                    # print('em= ', em)
                    return em #cm^-6 kpc
                    
                tup = (*zip(l.flatten(), b.flatten(), self.integrateTill.flatten()),)
                self.EM = np.array( (*map(_calc, tup),) ).reshape(l.shape) #cm^-6 kpc
        else:
            LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100) #np.logspace(-6, np.log10(integrateTill), 10)
            radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
            height = np.abs(radius*np.cos(np.deg2rad(theta)))
            radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                       
            ne = ne_prof(radius, height)
            ni = ni_prof(radius, height)
            
            self.EM = np.trapz( np.nan_to_num(ne*ni), LOSsample ) #cm^-6 kpc
        
        self.EM *= 1e3 #convert to cm^-6 pc
        return self.EM 