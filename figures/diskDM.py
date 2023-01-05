#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:58:36 2022

@author: alankar

Reference: Yamasaki and Totani 2020
"""

import sys
sys.path.append('..')
import numpy as np
from misc.ProgressBar import ProgressBar
from  misc.constants import *
import observable.CoordinateTrans as transform
import observable.maps as maps

class DispersionMeasureDisk(maps.MapInit):
    
    def __init__(self, redisProf):
        super().__init__(redisProf)
        self.redisProf = redisProf
            
    def generate(self, l, b, showProgress=True):
        l, b = super().prepare(l,b)
        rend = self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
        
            
        def ne_prof(R, z):
            n0 = 3.8e-3 # cm^-3
            R0 = 7.0 #kpc
            z0 = 2.7 #kpc
            return n0 * np.exp(-(R/R0+ np.fabs(z)/z0))

        if isinstance(l, np.ndarray):          
            if showProgress: 
                progBar = None
                self.DM = np.zeros_like(l)
                for i in range(self.DM.shape[0]):
                    for j in range(self.DM.shape[1]):
                        LOSsample = np.logspace(np.log10(1e-3*self.integrateTill[i,j]), 
                                                np.log10(self.integrateTill[i,j]), 100) 
                        radius, phi, theta = transform.toGalC(l[i,j], b[i,j], LOSsample)
                        height = np.abs(radius*np.cos(np.deg2rad(theta)))
                        radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                        
                        dm = np.trapz( np.nan_to_num(ne_prof(radius, height)) , LOSsample)
                        # print('dm= ', dm)
                        self.DM[i,j] =  dm #cm^-3 kpc
                        if (i==0 and j==0): progBar = ProgressBar()
                        progBar.progress(i*self.DM.shape[1]+j+1, 
                                         self.DM.shape[0]*self.DM.shape[1])
                progBar.end()
            else:
                def _calc(tup):
                    l_val, b_val, integrateTill = tup
                    LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100) #np.logspace(-6, np.log10(integrateTill), 10)
                    radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
                    height = np.abs(radius*np.cos(np.deg2rad(theta)))
                    radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                    # print('ne: ', np.nan_to_num(ne_prof(radius, height)) )
                    dm = np.trapz( np.nan_to_num(ne_prof(radius, height)), LOSsample)
                    # print('dm= ', dm)
                    return dm #cm^-3 kpc
                    
                tup = (*zip(l.flatten(), b.flatten(), self.integrateTill.flatten()),)
                self.DM = np.array( (*map(_calc, tup),) ).reshape(l.shape) #cm^-6 kpc
        else:
            LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100)
            radius, phi, theta = transform.toGalC(l, b, LOSsample)
            height = np.abs(radius*np.cos(np.deg2rad(theta)))
            radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
            self.DM = np.trapz( np.nan_to_num(ne_prof(radius, height)), LOSsample) #cm^-3 kpc
        
        self.DM *= 1e3 #convert to cm^-3 pc
        return self.DM 
    