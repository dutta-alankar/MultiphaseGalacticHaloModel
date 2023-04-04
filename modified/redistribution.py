#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:18:19 2023

@author: alankar
"""

import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.special import erf
from scipy.optimize import root
sys.path.append('..')
from misc.constants import mp, mH, kpc, km, s, K, kB, G, pi, Xp
from misc.coolLambda import cooling_approx
sys.path.append('..')
sys.path.append('../submodules/AstroPlasma')
from astro_plasma import Ionization

class Redistribution:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, unmodifiedProfile, TmedVW=3.e5, sig=0.3, cutoff=4.0):
        self.TmedVW = TmedVW
        self.sig    = sig   # spread of unmodified temperature redistribution
        self.cutoff = cutoff
        self.unmodified = unmodifiedProfile
        self.sigH   = self.sig
        self.sigW   = self.sig
        self.redshift = unmodifiedProfile.redshift
        self.ionization  = unmodifiedProfile.ionization
        
    def ProfileGen(self, radius_):  #takes in radius_ in kpc, returns Halo density and pressure  in CGS
        mu = Ionization().interpolate_mu
        if not(hasattr(self, "isobaric")):
            print("Error: Missing thermodynamic modification condition: isobaric/isochoric!")
        isobaric = self.isobaric
        
        radius = radius_*kpc/self.__class__.UNIT_LENGTH
        unmod_rho, unmod_prsTh, _, _, unmod_prsTot, unmod_nH, unmod_mu = self.unmodified.ProfileGen(radius_) #CGS
        self.metallicity = self.unmodified.metallicity
        unmod_T = (unmod_prsTh/kB)/(unmod_rho/(unmod_mu*mp))
        self.TmedVH = unmod_T
        # This changes even if condition is isobaric, but serves as a good guess for nH
        unmod_prsHTh = unmod_nH*kB*unmod_T 
        
        tdyn  = np.sqrt(radius**3/(\
               (G*self.__class__.UNIT_LENGTH**2*self.__class__.UNIT_DENSITY/self.__class__.UNIT_VELOCITY**2)\
                   *(self.unmodified.Halo.Mass(radius_)/self.__class__.UNIT_MASS))) # code
        tcool = lambda ndens, nH, Temp, met:((1.5+isobaric)*ndens*kB*Temp/(nH*nH*cooling_approx(Temp, met)))\
                /self.__class__.UNIT_TIME # ndens in CGS , tcool in code, 4.34 for FM17 LAMDBA norm
                
        # Warm gas
        Tstart = 3.9
        Tstop  = 6.5
        Temp  = np.logspace(Tstart, Tstop, 20) # Find tcool/tff for these temperature values
        if hasattr(self, "_plot_xval"):
           Temp = self._plot_xval
        fvw   = np.zeros_like(radius)
        fmw   = np.zeros_like(radius)
        Tcut  = np.zeros_like(radius)
        xmin  = np.zeros_like(radius)
        
        def _calc(r_val): # This always called for each memeber of radius array
            indx = np.argwhere(np.isclose(radius, r_val))[0,0] # index for the passed radius
            r_val = r_val*self.__class__.UNIT_LENGTH # CGS
            ndens = unmod_prsTh[indx]/(kB*Temp) # CGS
            nH_guess = unmod_prsHTh[indx]/(kB*Temp) # guess
                
            nH    = 10.**np.array([root(lambda LognH: (unmod_prsTh[indx]/kB)*Xp(self.metallicity[indx])* \
                           mu(10.**LognH, Temp[i] , self.metallicity[indx], self.redshift, self.ionization) - \
                           10.**LognH * Temp[i] * (mH/mp), np.log10(nH_guess[i])).x[0] for i in range(Temp.shape[0])])            
            
            unmod_tcool = tcool(ndens, nH, Temp, self.metallicity[indx]) #code
            ratio = unmod_tcool/tdyn[indx]
            
            if (self.cutoff>=0.1):
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                # cutoff in log T where seperation between hot and warm phases occur
                xmin[indx]  = np.log(Tmin/(unmod_T[indx]*np.exp(self.sigH**2/2))) 
                # xwarm = np.log(self.TmedVW/(unmod_T[indx]*np.exp(self.sigH**2/2)))
            else:
                # This part is anyways not physically important
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                xmin[indx] = -np.inf       
                
            fmw[indx] = 0.5*(1 + erf((xmin[indx]+self.sig**2)/(np.sqrt(2)*self.sig)))
            fvw[indx] = 0.5*(1 + erf(xmin[indx]/(np.sqrt(2)*self.sig)))   

        _ = (*map(_calc, radius),) # this is faster than just using for loop
       
        # Now calculate the relevant quantities for the modified profile
        self.Tcut = Tcut
        self.rhowarm_local = unmod_rho*(fmw/fvw)
        self.rhohot_local  = unmod_rho*((1-fmw)/(1-fvw)) 
        self.nHwarm_local  = self.rhowarm_local*Xp(self.metallicity)/mH
        self.nHhot_local   = self.rhohot_local*Xp(self.metallicity)/mH
        
        self.mu_warm = np.array([ mu(self.nHwarm_local[i], self.TmedVW*np.exp(-self.sigW**2/2), 
                                self.metallicity[i], self.redshift, self.ionization) for i in range(radius_.shape[0])])
        self.mu_hot  = np.array([ mu(self.nHhot_local[i], self.TmedVH[i]*np.exp(-self.sigH**2/2), 
                                self.metallicity[i], self.redshift, self.ionization) for i in range(radius_.shape[0])])
        
        self.nwarm_local  = self.rhowarm_local/(self.mu_warm*mp)
        self.nhot_local   = self.rhohot_local/(self.mu_hot*mp)
        
        self.nwarm_local  = np.piecewise(self.nwarm_local, [np.isnan(self.nwarm_local),], [lambda x:0, lambda x:x])
        self.nhot_local   = np.piecewise(self.nhot_local, [np.isnan(self.nhot_local),], [lambda x:0, lambda x:x])
        self.nwarm_global = self.nwarm_local*fvw
        self.nhot_global  = self.nhot_local*(1-fvw)
        self.prs_warm     = self.nwarm_local*kB*self.TmedVW*np.exp(-self.sigW**2/2)
        self.prs_hot      = self.nhot_local*kB*unmod_T
        self.radius       = radius_ #kpc
        self.fvw          = fvw
        self.fmw          = fmw
        self.TempDist     = Temp
        self.xmin         = xmin
        
        return (self.nhot_local, self.nwarm_local, self.nhot_global, self.nwarm_global, fvw, fmw, self.prs_hot, self.prs_warm, Tcut)
    
    def MassGen(self, radius_, isobaric): #takes in r in kpc, returns Halo gas mass of each component in CGS
        _call_profile = True # check if ProfileGen needs to be called
        if hasattr(self, "radius"): 
            if radius_ == self.radius:
                _call_profile = False     
        if (_call_profile):
            _ = self.ProfileGen(radius_, isobaric)
            
        MHot = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(self.nhot_global*self.mu_hot*mp), radius_*kpc)
        MWarm = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(self.nwarm_global*self.mu_warm*mp), radius_*kpc)
        
        return (MHot, MWarm)
    
    def PlotDistributionGen(self, radius, figure=None): # takes in radius in kpc (one value at any call)
        _call_profile = True # check if ProfileGen needs to be called
        if hasattr(self, "radius") and hasattr(self, "_plot_xval"): 
            if radius in self.radius:
                _call_profile = False     
        if (_call_profile):
            Tstart = 3.9
            Tstop  = 7.9
            Temp  = np.logspace(Tstart, Tstop, 1000) # Find tcool/tff for these temperature values
            self._plot_xval = Temp
            _ = self.ProfileGen(np.array([radius,]))
               
        import matplotlib.pyplot as plt    
        if (figure==None):    
            fig = plt.figure(figsize=(13,10))
        else:
            fig = figure
        self.fig = fig
        ax = fig.gca()
        
        x = np.log(self.TempDist/(self.TmedVH*np.exp(self.sigH**2/2)))
        gvh = np.exp(-x**2/(2*self.sigH**2))/(self.sigH*np.sqrt(2*pi))
        xp = np.log(self.TempDist/self.TmedVW)
        gvw = self.fvw*np.exp(-xp**2/(2*self.sigW**2))/(self.sigW*np.sqrt(2*pi))
        Tcutoff = np.exp(self.xmin)*(self.TmedVH*np.exp(self.sigH**2/2))
        
        plt.vlines(np.log10(Tcutoff), 1e-3, 2.1, colors='black', linestyles='--', label=r'$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$'%self.cutoff, 
                     linewidth=5, zorder=20)
        plt.vlines(np.log10(self.TmedVH*np.exp(self.sigH**2/2)), 1e-3, 2.1, colors='tab:red', linestyles=':', label=r'$T_{med,V}^{(h)}$', 
                     linewidth=5, zorder=30)
        plt.vlines(np.log10(self.TmedVW), 1e-3, 2.1, colors='tab:blue', linestyles=':', label=r'$T_{med,V}^{(w)}$', 
                     linewidth=5, zorder=40)
        
        plt.semilogy(np.log10(self.TempDist), 
                     np.piecewise(gvh,[self.TempDist>=Tcutoff,],[lambda val:val, lambda val:0]), 
                     color='tab:red', label='hot, modified', 
                     linewidth=5, zorder=5)
        plt.semilogy(np.log10(self.TempDist), gvh, color='tab:red', alpha=0.5, label='hot, unmodified', 
                     linewidth=5, zorder=6)
        plt.semilogy(np.log10(self.TempDist), gvw, color='tab:blue', label='warm', linestyle='--', 
                     linewidth=5, zorder=7)
        
        if (figure==None):          
            plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
            plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
            plt.grid()
            plt.title(r'$r = $%.1f kpc [isothermal with isochoric modification] (%s)'%(radius, self.ionization), size=28)
            plt.ylim(1e-3, 2.1)
            plt.xlim(5, 7)
            plt.ylabel(r'$T \mathscr{P}_V(T)$', size=28)
            plt.xlabel(r'$\log_{10} (T [K])$', size=28)
            # ax.yaxis.set_ticks_position('both')
            plt.legend(loc='upper right', prop={'size': 20}, framealpha=0.3, shadow=False, fancybox=True, bbox_to_anchor=(1.1, 1))
            plt.savefig('isothermal_isochoric_PDF_%s.png'%self.ionization, transparent=True)
            plt.show()
