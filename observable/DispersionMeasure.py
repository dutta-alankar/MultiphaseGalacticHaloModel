#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 10:22:13 2022

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root
from misc.ProgressBar import ProgressBar
from  misc.constants import *
import observable.CoordinateTrans as transform
import observable.maps as maps
from misc.ionization import interpolate_ionization

class DispersionMeasure(maps.MapInit):
    
    def __init__(self, redisProf):
        super().__init__(redisProf)
        self.redisProf = redisProf
        self.genProf = False
        
    def ne_gen(self, distance):
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        
        ionization = interpolate_ionization()
        num_dens = lambda tup: ionization.num_dens(*tup)
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        # mu for 100% ionization and Solar for quick t_cool calculation as mu doesn't change a lot for ranges of temperature and density of interest
        #mu = 1./(2*Xp+(3/4)*Yp+(9/16)*Zp)
        
        # redistributed profile is generated for only a limited number of points and used for interpolation
        if not(self.genProf):
            rend = 1.01*self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
            print("Doing one time profile calculation", flush=True)
            radius_ = np.logspace(np.log10(5.0), np.log10(rend),20) #kpc
            nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = self.redisProf.ProfileGen(radius_)
            self.genProf = True
            # np.save('mod-debug.npy', np.vstack((radius_, nhot_local, nwarm_local, prs_hot/(nhot_local*kB), prs_warm/(nwarm_local*kB) )).T ) 
            # print(nhot_local, nwarm_local)
            print('Complete!', flush=True)
            
        mu = interpolate_ionization().mu
        
        nHhot_local  = interp1d( self.redisProf.radius, self.redisProf.nHhot_local, fill_value='extrapolate')
        nHwarm_local = interp1d( self.redisProf.radius, self.redisProf.nHwarm_local, fill_value='extrapolate')
        prs_hot      = interp1d( self.redisProf.radius, self.redisProf.prs_hot, fill_value='extrapolate')
        prs_warm     = interp1d( self.redisProf.radius, self.redisProf.prs_warm, fill_value='extrapolate')
        Tcut         = interp1d( self.redisProf.radius, self.redisProf.Tcut, fill_value='extrapolate')
        metallicity  = interp1d( self.redisProf.radius, self.redisProf.unmodified.metallicity, fill_value='extrapolate')
        fvw          = interp1d( self.redisProf.radius, self.redisProf.fvw, fill_value='extrapolate')
        TmedVH       = interp1d( self.redisProf.radius, self.redisProf.TmedVH, fill_value='extrapolate')
                
        Temp  = self.redisProf.TempDist
        
        THotM  = interp1d( self.redisProf.radius, (self.redisProf.prs_hot/(self.redisProf.nhot_local*kB)), fill_value='extrapolate')
        # THotM  = interp1d( radius_, (self.redisProf.prs_hot/self.redisProf.rhohot_local)*(kB/(self.redisProf.mu_hot*mp)), fill_value='extrapolate')
        # TWarmM = interp1d( self.redisProf.radius, (self.redisProf.prs_warm/self.redisProf.rhowarm_local)*(kB/(self.redisProf.mu_warm*mp)), fill_value='extrapolate')
        
        def _calc(r_val):    
            xh = np.log(Temp/(THotM(r_val)*np.exp(self.redisProf.sigH**2/2)))
            PvhT = np.exp(-xh**2/(2*self.redisProf.sigH**2))/(self.redisProf.sigH*np.sqrt(2*pi))
            xw = np.log(Temp/self.redisProf.TmedVW)
            gvwT = fvw(r_val)*np.exp(-xw**2/(2*self.redisProf.sigW**2))/(self.redisProf.sigW*np.sqrt(2*pi))
            gvhT = np.piecewise(PvhT, [Temp>=Tcut(r_val),], [lambda xp:xp, lambda xp:0.])
            
            #Assumtion: Phases are internally isobaric
            ndensHot  = prs_hot(r_val)/(kB*Temp) #CGS
            ndensWarm = prs_warm(r_val)/(kB*Temp) #CGS
            
            # Approximation is nH T is also constant like n T used as guess
            nHhot_guess  = nHhot_local(r_val)*TmedVH(r_val)*np.exp(-self.redisProf.sigH**2/2)/Temp #CGS
            nHwarm_guess = nHwarm_local(r_val)*self.redisProf.TmedVW*np.exp(-self.redisProf.sigW**2/2)/Temp #CGS
            
            nHhot  = 10.**np.array([root( lambda LognH: (prs_hot(r_val)/(kB*Temp[i])) * Xp(metallicity(r_val))*\
                                    mu(10**LognH, Temp[i], metallicity(r_val), redshift, mode) - \
                                    (mH/mp)*(10**LognH), np.log10(nHhot_guess[i]) ).x[0] for i in range(Temp.shape[0])])
            nHwarm = 10.**np.array([root( lambda LognH: (prs_warm(r_val)/(kB*Temp[i])) * Xp(metallicity(r_val))*\
                                    mu(10**LognH, Temp[i], metallicity(r_val), redshift, mode) - \
                                    (mH/mp)*(10**LognH), np.log10(nHwarm_guess[i]) ).x[0] for i in range(Temp.shape[0])])
            
            tup = (*zip( nHhot, Temp, 
                         metallicity(r_val)*np.ones(Temp.shape[0]), 
                         redshift*np.ones(Temp.shape[0]),
                         [mode,]*Temp.shape[0], ['electron',]*Temp.shape[0] ),)
            
            neHot = np.array( (*map(num_dens, tup),) ).reshape(Temp.shape)
            
            tup = (*zip( nHwarm, Temp, 
                         metallicity(r_val)*np.ones(Temp.shape[0]), 
                         redshift*np.ones(Temp.shape[0]),
                         [mode,]*Temp.shape[0], ['electron',]*Temp.shape[0] ),)
            
            neWarm = np.array( (*map(num_dens, tup),) ).reshape(Temp.shape)
            
            '''
            neHot  = np.array([num_dens(     nHhot[i], 
                                             Temp[i], 
                                             metallicity(r_val), redshift, 
                                             mode=mode, part_type='electron' ) for i,xhp in enumerate(xh)])
            
            neWarm = np.array([num_dens(     nHwarm[i], 
                                             Temp[i], 
                                             metallicity(r_val), redshift, 
                                             mode=mode, part_type='electron' ) for i,xwp in enumerate(xw)])
            '''
            hotInt  = (1-fvw(r_val))*np.trapz( neHot*gvhT, xh ) #global density sensitive, extra filling factor for global 
        
            warmInt = fvw(r_val)*np.trapz( neWarm*gvwT, xw )
            
            ne = hotInt + warmInt
            return ne
        
        ne = np.array((*map(_calc, distance),)) 
        # ne = np.piecewise(ne, [ne>=1e-6,], [lambda x:x, lambda x:0.] )
        return ne
            
    def generate(self, l, b, showProgress=True):
        l, b = super().prepare(l,b)
        rend = self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
        
        distance = np.logspace(np.log10(5.0), 1.01*np.log10(rend), 20) #kpc
        print('Generating free electron profile')
        ne_prof = self.ne_gen(distance)
        print('Complete!')
        print('ini: rad: ', distance)
        print('ini: ne:  ', ne_prof)
        np.save('test-debug-ne.npy', np.vstack((distance, ne_prof)).T )
        ne_prof  = interp1d( np.log10(distance), np.log10(ne_prof), fill_value='extrapolate')
        if isinstance(l, np.ndarray):          
            if showProgress: 
                progBar = None
                self.DM = np.zeros_like(l)
                for i in range(self.DM.shape[0]):
                    for j in range(self.DM.shape[1]):
                        LOSsample = np.logspace(np.log10(1e-3*self.integrateTill[i,j]), np.log10(self.integrateTill[i,j]), 100) #np.logspace(-6, np.log10(self.integrateTill[i,j]), 10) # points on the LOS
                        radius, phi, theta = transform.toGalC(l[i,j], b[i,j], LOSsample)
                        height = np.abs(radius*np.cos(np.deg2rad(theta)))
                        radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
                        distance = np.sqrt(radius**2 + height**2)
                        # print('rad: ', distance)
                        # print('ne: ', np.nan_to_num(10.**ne_prof(np.log10(distance))) )
                        dm= np.trapz( np.nan_to_num(10.**ne_prof(np.log10(distance))) , LOSsample)
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
                    distance = np.sqrt(radius**2 + height**2)
                    # print('rad: ', distance)
                    # print('ne: ', np.nan_to_num(10.**ne_prof(np.log10(distance))) )
                    dm = np.trapz( np.nan_to_num(10.**ne_prof(np.log10(distance))), LOSsample)
                    # print('dm= ', dm)
                    return dm #cm^-3 kpc
                    
                tup = (*zip(l.flatten(), b.flatten(), self.integrateTill.flatten()),)
                self.DM = np.array( (*map(_calc, tup),) ).reshape(l.shape) #cm^-6 kpc
        else:
            LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100)
            radius, phi, theta = transform.toGalC(l, b, LOSsample)
            height = np.abs(radius*np.cos(np.deg2rad(theta)))
            radius = np.abs(radius*np.sin(np.deg2rad(theta))) #np.cos(np.deg2rad(phi))) #along disk
            distance = np.sqrt(radius**2 + height**2)
            self.DM = np.trapz( np.nan_to_num(10.**ne_prof(np.log10(distance))), LOSsample) #cm^-3 kpc
        
        self.DM *= 1e3 #convert to cm^-3 pc
        return self.DM 