# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:28:36 2022

@author: Alankar
"""

import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.special import erf
from scipy.optimize import root
sys.path.append('..')
from misc.constants import *
from misc.HaloModel import HaloModel
from misc.coolLambda import cooling_approx
from misc.ionization import interpolate_ionization

class IsobarCoolRedistribution:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    def __init__(self, unmodifiedProfile, TmedVW=3.e5, sig=0.3, cutoff=2):
        self.TmedVW = TmedVW
        self.sig    = sig   # spread of unmodified temperature redistribution
        self.cutoff = cutoff
        self.unmodified = unmodifiedProfile
        self.sigH   = self.sig
        self.sigW   = self.sig
        self._plot  = False
        self.redshift = unmodifiedProfile.redshift
        self.ionization  = unmodifiedProfile.ionization
        
    def ProfileGen(self, radius_):  #takes in radius_ in kpc, returns Halo density and pressure  in CGS
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        # mu for 100% ionization and Solar for quick t_cool calculation as mu doesn't change a lot for ranges of temperature and density of interest
        #mu = 1./(2*Xp+(3/4)*Yp+(9/16)*Zp)
        mu = interpolate_ionization().mu
        
        radius = radius_*kpc/IsobarCoolRedistribution.UNIT_LENGTH
        unmod_rho, unmod_prsTh, _, _, unmod_prsTot, unmod_nH, unmod_mu = self.unmodified.ProfileGen(radius_) #CGS
        self.metallicity = self.unmodified.metallicity
        unmod_T = (unmod_prsTh/kB)/(unmod_rho/(unmod_mu*mp))
        self.TmedVH = unmod_T
        
        unmod_n  = unmod_prsTh/(kB*unmod_T)  
        unmod_prsHTh = unmod_nH*kB*unmod_T
        
        isobaric = 1
        tdyn  = np.sqrt(radius**3/(\
               (G*IsobarCoolRedistribution.UNIT_LENGTH**2*IsobarCoolRedistribution.UNIT_DENSITY/IsobarCoolRedistribution.UNIT_VELOCITY**2)\
                   *(self.unmodified.Halo.Mass(radius_)/IsobarCoolRedistribution.UNIT_MASS))) #code
        tcool = lambda ndens, nH, Temp, met:((1.5+isobaric)*ndens*kB*Temp/(nH*nH*cooling_approx(Temp, met)))\
                /IsobarCoolRedistribution.UNIT_TIME #ndens in CGS , tcool in code, 4.34 for FM17 LAMDBA norm   
             
        
        #Warm gas
        Tstart = 4.1
        Tstop  = 7.9
        Temp  = np.logspace(Tstart, Tstop, 20) #find tcool/tff for these temperature values
        fvw   = np.zeros_like(radius)
        fmw   = np.zeros_like(radius)
        Tcut  = np.zeros_like(radius)
        
        for indx, r_val in enumerate(radius):
            r_val = r_val*IsobarCoolRedistribution.UNIT_LENGTH #CGS
            ndens = unmod_prsTh[indx]/(kB*Temp) #CGS
            # nH    = unmod_prsHTh[indx]/(kB*Temp) # guess
            nH    = 10.**np.array([root(lambda LognH: (unmod_prsTh[indx]/kB)*Xp(self.metallicity[indx])* \
                           mu(10.**LognH, Temp[i] , self.metallicity[indx], self.redshift, self.ionization) - \
                           10.**LognH * Temp[i] * (mH/mp), np.log10(ndens[i])).x[0] for i in range(Temp.shape[0])])
            # nH    = unmod_prsHTh[indx]/(kB*Temp)
            # 10.**np.array([root(lambda LognH: 10.**LognH - 
            #                     ndens[indx]*Xp(self.metallicity[indx])*(mp/mH)*
            #                     (interpolate_ionization().mu(10.**LognH, Temp[i] , self.metallicity[indx], self.redshift, self.ionization)), -3).x[0] for i in range(Temp.shape[0])]
            unmod_tcool = tcool(ndens, nH, Temp, self.metallicity[indx]) #code
            ratio = unmod_tcool/tdyn[indx]
            
            if (self.cutoff>=0.1):
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                xmin  = np.log(Tmin/(unmod_T[indx]*np.exp(self.sigH**2/2))) #cutoff in log T where seperation between hot and warm phases occur
                xwarm = np.log(self.TmedVW/(unmod_T[indx]*np.exp(self.sigH**2/2)))
            else:
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                xmin = -np.inf
                
            #if (Tmin>unmod_T[indx]): print('Trouble!') #xmin<0
            if (True): #(xmin-xwarm)>self.sigW/2):#): #unmod_T[indx]>self.TmedVW or 
                
                #xmin = np.log(Tmin/self.TmedVH) #cutoff in log T where seperation between hot and warm phases occur
                fmw[indx] = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))
                fvw[indx] = (self.TmedVW/unmod_T[indx])*np.exp(-0.5*self.sigW**2)*fmw[indx]
            else:
                pass
                
        '''
        if (self._plot):
            import matplotlib.pyplot as plt
            #print(ratio)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(radius_, Tcut/1e5)
            plt.plot(radius_, self.TmedVW*np.ones_like(radius_)/1e5, linestyle=':', color='gray')
            plt.plot(radius_, (10.**(self.sigW*0.5))*self.TmedVW*np.ones_like(radius_)/1e5, linestyle=':', color='black')
            plt.grid()
            plt.ylim(ymin=2, ymax=7)
            plt.xlabel(r'$r$ [kpc]')
            plt.ylabel(r'Temperature [K] $\times 1e5$)')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        '''
        #epsilon  = 1e-6
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
        
        # self.metallicity = interpolate.interp1d(radius_, self.metallicity, fill_value="extrapolate")
        return (self.nhot_local, self.nwarm_local, self.nhot_global, self.nwarm_global, fvw, fmw, self.prs_hot, self.prs_warm, Tcut)
    
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
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        # mu for 100% ionization and Solar for quick t_cool calculation as mu doesn't change a lot for ranges of temperature and density of interest
        #mu = 1./(2*Xp+(3/4)*Yp+(9/16)*Zp)
        mu = interpolate_ionization().mu
        
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
            
        radius = np.copy(radius_)*kpc/IsobarCoolRedistribution.UNIT_LENGTH
        unmod_rho, unmod_prsTh, _, _, unmod_prsTot, unmod_nH, unmod_mu = self.unmodified.ProfileGen(radius_) #CGS
        metallicity = self.unmodified.metallicity
        unmod_T = (unmod_prsTh/kB)/(unmod_rho/(unmod_mu*mp))
        
        unmod_n  = unmod_prsTh/(kB*unmod_T)  
        unmod_prsHTh = unmod_nH*kB*unmod_T
        
        isobaric = 1
        tdyn  = np.sqrt(radius**3/(\
               (G*IsobarCoolRedistribution.UNIT_LENGTH**2*IsobarCoolRedistribution.UNIT_DENSITY/IsobarCoolRedistribution.UNIT_VELOCITY**2)\
                   *(self.unmodified.Halo.Mass(radius_)/IsobarCoolRedistribution.UNIT_MASS))) #code
        tcool = lambda ndens, nH, Temp, met:((1.5+isobaric)*ndens*kB*Temp/(nH*nH*cooling_approx(Temp, met)))\
                /IsobarCoolRedistribution.UNIT_TIME #ndens in CGS , tcool in code
                        
        #Warm gas
        Tstart = 4.1
        Tstop  = 7.9
        Temp  = np.logspace(Tstart, Tstop, 1000) #find tcool/tff for these temperature values
        fvw   = np.zeros_like(radius)
        fmw   = np.zeros_like(radius)
        Tcut  = np.zeros_like(radius)
        
        for indx, r_val in enumerate(radius):
            r_val = r_val*IsobarCoolRedistribution.UNIT_LENGTH #CGS
            ndens = unmod_prsTh[indx]/(kB*Temp) #CGS
            nH    = unmod_prsHTh[indx]/(kB*Temp) # guess
            nH    = 10.**np.array([root(lambda LognH: (unmod_prsTh[indx]/kB)*Xp(self.metallicity[indx])* \
                           mu(10.**LognH, Temp[i] , self.metallicity[indx], self.redshift, self.ionization) - \
                           10.**LognH * Temp[i] * (mH/mp), np.log10(nH[i])).x[0] for i in range(Temp.shape[0])])
            # nH    = unmod_prsHTh[indx]/(kB*Temp)
            # 10.**np.array([root(lambda LognH: 10.**LognH - 
            #                     ndens[indx]*Xp(self.metallicity[indx])*(mp/mH)*
            #                     (interpolate_ionization().mu(10.**LognH, Temp[i] , self.metallicity[indx], self.redshift, self.ionization)), -3).x[0] for i in range(Temp.shape[0])]
            unmod_tcool = tcool(ndens, nH, Temp, self.metallicity[indx]) #code
            ratio = unmod_tcool/tdyn[indx]
            
            if (self.cutoff>=0.1):
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                xmin  = np.log(Tmin/(unmod_T[indx]*np.exp(self.sigH**2/2))) #cutoff in log T where seperation between hot and warm phases occur
                xwarm = np.log(self.TmedVW/(unmod_T[indx]*np.exp(self.sigH**2/2)))
            else:
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff) 
                Tcut[indx] = Tmin
                xmin = -np.inf
                
            #if (Tmin>unmod_T[indx]): print('Trouble!') #xmin<0
            if (True): #(xmin-xwarm)>self.sigW/2):#): #unmod_T[indx]>self.TmedVW or 
                
                #xmin = np.log(Tmin/self.TmedVH) #cutoff in log T where seperation between hot and warm phases occur
                fmw[indx] = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))
                fvw[indx] = (self.TmedVW/unmod_T[indx])*np.exp(-0.5*self.sigW**2)*fmw[indx]
            else:
                pass
                
            fig = plt.figure(figsize=(13,10))
            ax = fig.add_subplot(111)
            x = np.log(Temp/(unmod_T*np.exp(self.sigH**2/2)))
            gvh = np.exp(-x**2/(2*self.sigH**2))/(self.sigH*np.sqrt(2*pi))
            xp = np.log(Temp/self.TmedVW)
            gvw = fvw*np.exp(-xp**2/(2*self.sigW**2))/(self.sigW*np.sqrt(2*pi))
            Tcutoff = np.exp(xmin)*(unmod_T*np.exp(self.sigH**2/2))
            
            plt.vlines(np.log10(Tcutoff), 1e-3, 2.1, colors='black', linestyles='--', label=r'$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$'%self.cutoff, 
                         linewidth=5, zorder=20)
            plt.vlines(np.log10(unmod_T*np.exp(self.sigH**2/2)), 1e-3, 2.1, colors='tab:red', linestyles=':', label=r'$T_{med,V}^{(h)}$', 
                         linewidth=5, zorder=30)
            plt.vlines(np.log10(self.TmedVW), 1e-3, 2.1, colors='tab:blue', linestyles=':', label=r'$T_{med,V}^{(w)}$', 
                         linewidth=5, zorder=40)
            
            plt.semilogy(np.log10(Temp), np.piecewise(gvh,[Temp>=Tcutoff,],[lambda val:val, lambda val:0]), color='tab:red', label='hot, modified', 
                         linewidth=5, zorder=5)
            plt.semilogy(np.log10(Temp), gvh, color='tab:red', alpha=0.5, label='hot, unmodified', 
                         linewidth=5, zorder=6)
            plt.semilogy(np.log10(Temp), gvw, color='tab:blue', label='warm', linestyle='--', 
                         linewidth=5, zorder=7)
            #plt.semilogx(Temp, ratio, label='tcool/tdyn=%d'%cutoff, color='tab:gray')
                      
            plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
            plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
            plt.grid()
            plt.title(r'$r = $%.1f kpc [isothermal with isobaric modification] (%s)'%(radius_, self.ionization), size=28)
            plt.ylim(1e-3, 2.1)
            plt.xlim(5, 7)
            plt.ylabel(r'$T \mathscr{P}_V(T)$', size=28)
            plt.xlabel(r'$\log_{10} (T [K])$', size=28)
            # ax.yaxis.set_ticks_position('both')
            plt.legend(loc='upper right', prop={'size': 20}, framealpha=0.3, shadow=False, fancybox=True, bbox_to_anchor=(1.1, 1))
            plt.savefig('isothermal_isobaric_PDF_%s.png'%self.ionization, transparent=True)
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
