# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:15:09 2022

@author: Alankar
"""

import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.special import erf
sys.path.append('..')
from misc.constants import *
from misc.HaloModel import HaloModel

class IsochorCoolRedistribution:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, unmodifiedProfile, TmedVH=1.5e6, TmedVW=3.e5, sig=0.3, cutoff=2):
        self.TmedVH = TmedVH
        self.TmedVW = TmedVW
        self.sig    = sig   # spread of unmodified temperature redistribution
        self.cutoff = cutoff
        self.unmodified = unmodifiedProfile
        self.sigH   = self.sig
        self.sigW   = self.sig
        self._plot  = False
        
    def ProfileGen(self, radius_):  #takes in radius_ in kpc, returns Halo density and pressure  in CGS
        radius = radius_*kpc/IsochorCoolRedistribution.UNIT_LENGTH
        unmod_rho, unmod_prsTh, _, _, unmod_prsTot = self.unmodified.ProfileGen(radius_) #CGS
        unmod_n = unmod_rho/(mu*mp) #CGS
        unmod_T = unmod_prsTh/(unmod_n*kB) # mass avg temperature for unmodified profile
        # print('unmod T')
        # print(unmod_T)
        
        #LAMBDA = np.loadtxt('cooltable.dat')
        LAMBDA = 10**np.loadtxt('hazy_coolingcurve_0.5.txt', skiprows=1)
        LAMBDA = interpolate.interp1d(LAMBDA[:,0], LAMBDA[:,1])
        
        isobaric = 0
        tdyn  = np.sqrt(radius**3/(\
               (G*IsochorCoolRedistribution.UNIT_LENGTH**2*IsochorCoolRedistribution.UNIT_DENSITY/IsochorCoolRedistribution.UNIT_VELOCITY**2)\
                   *(self.unmodified.Halo.Mass(radius_)/IsochorCoolRedistribution.UNIT_MASS))) #code
        tcool = lambda ndens, Temp:(4.34*(1.5+isobaric)*kB*Temp/(ndens*LAMBDA(Temp)))\
                /IsochorCoolRedistribution.UNIT_TIME #ndens in CGS , tcool in code, 4.34 for FM17 LAMDBA norm
        
        #Warm gas
        Tstart = 4.1
        Tstop  = 7.9
        Temp  = np.logspace(Tstart, Tstop, 400) #find tcool/tff for these temperature values
        fvw   = np.zeros_like(radius)
        fmw   = np.zeros_like(radius)
        Tcut  = np.zeros_like(radius)

        for indx, r_val in enumerate(radius):
            r_val = r_val*IsochorCoolRedistribution.UNIT_LENGTH #CGS
            ndens = unmod_prsTh[indx]/(kB*Temp) #CGS
            unmod_tcool = tcool(ndens, Temp) #code
            ratio = unmod_tcool/tdyn[indx]
            
            Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
            Tmin = Tmin(self.cutoff) 
            Tcut[indx] = Tmin
            xmin  = np.log(Tmin/(unmod_T[indx]*np.exp(self.sigH**2/2))) #cutoff in log T where seperation between hot and warm phases occur
            xwarm = np.log(self.TmedVW/(unmod_T[indx]*np.exp(self.sigH**2/2)))
            
            #if (Tmin>unmod_T[indx]): print('Trouble!') #xmin<0
            if (True): #(xmin-xwarm)>self.sigW/2):#): #unmod_T[indx]>self.TmedVW or 
                
                #xmin = np.log(Tmin/self.TmedVH) #cutoff in log T where seperation between hot and warm phases occur
                fvw[indx] = 0.5*(1 + erf(xmin/(np.sqrt(2)*self.sig)))
                fmw[indx] = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))
            else:
                fvw[indx] = 0.5*(1 + erf(xmin/(np.sqrt(2)*self.sig)))
                fmw[indx] = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))

        '''
        if (self._plot):
            import matplotlib.pyplot as plt
            #print(ratio)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(radius_, Tcut/1e5)
            plt.plot(radius_, self.TmedVW*np.ones_like(radius_)/1e5, linestyle=':', color='gray')
            plt.plot(radius_, (10**(self.sigW*0.5))*self.TmedVW*np.ones_like(radius_)/1e5, linestyle=':', color='black')
            plt.grid()
            plt.ylim(ymin=2, ymax=7)
            plt.xlabel(r'$r$ [kpc]')
            plt.ylabel(r'Temperature [K] $\times 1e5$)')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        '''
        nwarm_local = unmod_n*(fmw/fvw)
        nhot_local  = unmod_n*((1-fmw)/(1-fvw))   
        nwarm_global = nwarm_local*fvw
        nhot_global  = nhot_local*(1-fvw)
        prs_warm = nwarm_local*kB*self.TmedVW*np.exp(-self.sigW**2/2)
        prs_hot  = nhot_local*kB*unmod_T
        return (nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut)
    
    def MassGen(self, radius_): #takes in r in kpc, returns Halo gas mass of each component in CGS
        
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.linspace(self.unmodified.Halo.r0*self.unmodified.Halo.UNIT_LENGTH/kpc, radius_, 200)
            _, _, nhot_global, nwarm_global, _, _, _, _, _ = self.ProfileGen(radius_)
            
            MHot = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(nhot_global*mu*mp), radius_*kpc)[-1]
            MWarm = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(nwarm_global*mu*mp), radius_*kpc)[-1]
        else:
            _, _, nhot_global, nwarm_global, _, _, _, _, _ = self.ProfileGen(radius_)
            MHot = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(nhot_global*mu*mp), radius_*kpc)
            MWarm = integrate.cumtrapz(4*pi*(radius_*kpc)**2*(nwarm_global*mu*mp), radius_*kpc)
        
        return (MHot, MWarm)
    
    def PlotDistributionGen(self, radius_): #takes in radius_ in kpc
        import matplotlib.pyplot as plt
        
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        
        for radius in radius_:
            radius = radius_*kpc/IsochorCoolRedistribution.UNIT_LENGTH
            unmod_rho, unmod_prsTh, _, _, unmod_prsTot = self.unmodified.ProfileGen(radius_) #CGS
            unmod_n = unmod_rho/(mu*mp) #CGS
            unmod_T = unmod_prsTh/(unmod_n*kB) # mass avg temperature for unmodified profile
            # print('unmod T')
            # print(unmod_T)
        
            #LAMBDA = np.loadtxt('cooltable.dat')
            LAMBDA = 10**np.loadtxt('hazy_coolingcurve_0.5.txt', skiprows=1)
            LAMBDA = interpolate.interp1d(LAMBDA[:,0], LAMBDA[:,1])
        
            isobaric = 0
            tdyn  = np.sqrt(radius**3/(\
                    (G*IsochorCoolRedistribution.UNIT_LENGTH**2*IsochorCoolRedistribution.UNIT_DENSITY/IsochorCoolRedistribution.UNIT_VELOCITY**2)\
                   *(self.unmodified.Halo.Mass(radius_)/IsochorCoolRedistribution.UNIT_MASS))) #code
            tcool = lambda ndens, Temp:(4.34*(1.5+isobaric)*kB*Temp/(ndens*LAMBDA(Temp)))\
                   /IsochorCoolRedistribution.UNIT_TIME #ndens in CGS , tcool in code, 4.34 for FM17 LAMDBA norm
        
            #Warm gas
            Tstart = 4.5
            Tstop  = 7.5
            Temp  = np.logspace(Tstart, Tstop, 400) #find tcool/tff for these temperature values
            
            ndens = unmod_prsTh/(kB*Temp) #CGS
            unmod_tcool = tcool(ndens, Temp) #code
            ratio = unmod_tcool/tdyn
            
            Tmin = interpolate.interp1d(ratio, Temp)
            Tmin = Tmin(self.cutoff) 
            xmin  = np.log(Tmin/(unmod_T*np.exp(self.sigH**2/2))) #cutoff in log T where seperation between hot and warm phases occur
            
            fvw = 0.5*(1 + erf(xmin/(np.sqrt(2)*self.sig)))
            fmw = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.log(Temp/(unmod_T*np.exp(self.sigH**2/2)))
            gvh = np.exp(-x**2/(2*self.sigH**2))/(self.sigH*np.sqrt(2*pi))
            xp = np.log(Temp/self.TmedVW)
            gvw = fvw*np.exp(-xp**2/(2*self.sigW**2))/(self.sigW*np.sqrt(2*pi))
            plt.semilogy(np.log10(Temp), gvh, color='tab:red', label='hot')
            plt.semilogy(np.log10(Temp), gvw, color='tab:blue', label='warm', linestyle='--')
            #plt.semilogx(Temp, ratio, label='tcool/tdyn=%d'%cutoff, color='tab:gray')
            Tcutoff = np.exp(xmin)*(unmod_T*np.exp(self.sigH**2/2))
            plt.vlines(np.log10(Tcutoff), 1e-3, 2.1, colors='black', linestyles='--', label=r'$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$'%self.cutoff)
            plt.vlines(np.log10(unmod_T*np.exp(self.sigH**2/2)), 1e-3, 2.1, colors='tab:red', linestyles=':', label=r'$T_{med,V}^{(h)}$')
            plt.vlines(np.log10(self.TmedVW), 1e-3, 2.1, colors='tab:blue', linestyles=':', label=r'$T_{med,V}^{(w)}$')
            #plt.semilogx(Temp, ratio)
            plt.grid()
            plt.title(r'$r =$ %.1f kpc [isothermal with isochoric modification]'%radius_)
            plt.ylim(1e-3, 2.1)
            plt.xlim(5, 7)
            plt.ylabel(r'$T \mathscr{P}_V(T)$')
            plt.xlabel(r'$\log_{10} (T [K])$')
            ax.yaxis.set_ticks_position('both')
            plt.legend(loc='best')
            plt.show()
            
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # x = np.log(Temp/(unmod_T*np.exp(self.sigH**2/2)))
            # gvh = np.exp(-x**2/(2*self.sigH**2))/(self.sigH*np.sqrt(2*pi))
            # xp = np.log(Temp/self.TmedVW)
            # gvw = fvw*np.exp(-xp**2/(2*self.sigW**2))/(self.sigW*np.sqrt(2*pi))
            # gvh = np.piecewise(gvh, [x>=xmin,], [lambda xpp:xpp, lambda xpp:0.])
            # plt.semilogx(Temp, gvh, color='tab:red', label='hot')
            # plt.semilogx(Temp, gvw, color='tab:blue', label='warm')
            # #plt.semilogx(Temp, ratio, label='tcool/tdyn=%d'%cutoff, color='tab:gray')
            # Tcutoff = np.exp(xmin)*(unmod_T*np.exp(self.sigH**2/2))
            # #plt.vlines(Tcutoff, -1, 1.4, colors='black', linestyles='--', label=r'$\rm T_{min}\ (T_{cool}/t_{dyn}=%.1f)$'%self.cutoff)
            # #plt.vlines((unmod_T*np.exp(self.sigH**2/2)), -1, 1.4, colors='tab:red', linestyles=':', label=r'$\rm T_{med,V,hot}$')
            # #plt.vlines(self.TmedVW, -1, 1.4, colors='tab:blue', linestyles=':', label=r'$\rm T_{med,V,warm}$')
            # #plt.semilogx(Temp, ratio)
            # plt.grid()
            # plt.title('r = %.1f kpc [Modified]'%radius_)
            # plt.ylim(0., 1.4)
            # plt.ylabel(r'$Tp_{v}(T)$')
            # plt.xlabel('Temperature [K]')
            # ax.yaxis.set_ticks_position('both')
            # plt.legend(loc='best')
            # plt.show()
