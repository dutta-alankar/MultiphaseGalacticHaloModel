# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:53:50 2022

@author: Alankar
"""

import sys
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import root
sys.path.append('..')
from misc.constants import *
from misc.ionization import interpolate_ionization

class ColumnDensity:
    
    def __init__(self, redisProf):
        self.redisProf = redisProf
        self.genProf = False
        
    def ColumnDensityGen(self, b_, element=8, ion=6): 
        # takes in b_ in kpc, returns col dens in CGS
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        metallicity = None
        if isinstance(b_, float) or isinstance(b_, int):
            b_ = np.array([b_])
            #metallicity = np.array([metallicity_])
        #else: 
        #    metallicity = metallicity_
        
        '''
        frac = np.loadtxt('ion-frac-Oxygen.txt', skiprows=1, 
                          converters={i+1: lambda x: -np.inf if x==b'--' else x for i in range(element+1)})
        fOVII = interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
        '''
        ionFrac = interpolate_ionization()
        fIon = ionFrac.interpolate
        mu = interpolate_ionization().mu
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        # mu for 100% ionization and Solar for quick t_cool calculation as mu doesn't change a lot for ranges of temperature and density of interest
        #mu = 1./(2*Xp+(3/4)*Yp+(9/16)*Zp)
        
        #Use low initial resolution for interpolation
        if not(self.genProf):
            rend = 1.01*self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
            print("Doing one time profile calculation", flush=True)
            radius_ = np.logspace(np.log10(5.0), np.log10(rend),20) #kpc
            nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = self.redisProf.ProfileGen(radius_)
            self.genProf = True
            # np.save('mod-debug.npy', np.vstack((radius_, nhot_local, nwarm_local, prs_hot/(nhot_local*kB), prs_warm/(nwarm_local*kB) )).T ) 
            # print(nhot_local, nwarm_local)
            print('Complete!', flush=True)
        
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
        
        file_path = os.path.realpath(__file__)
        dir_loc   = os.path.split(file_path)[:-1]
        abn_file  = os.path.join(*dir_loc,'..','misc','cloudy-data', 'solar_GASS10.abn')
        _tmp = None
        with open(abn_file, 'r') as file:
            _tmp = file.readlines()
            
        abn = np.array([ float(element.split()[-1]) for element in _tmp[2:32] ]) #till Zinc
        a0  = abn[element-1]
        
        radius = self.redisProf.radius
        nIon = np.zeros_like(radius)
        for indx, r_val in enumerate(radius) :    
            xh = np.log(Temp/(THotM(r_val)*np.exp(self.redisProf.sigH**2/2)))
            PvhT = np.exp(-xh**2/(2*self.redisProf.sigH**2))/(self.redisProf.sigH*np.sqrt(2*pi))
            xw = np.log(Temp/self.redisProf.TmedVW)
            gvwT = fvw(r_val)*np.exp(-xw**2/(2*self.redisProf.sigW**2))/(self.redisProf.sigW*np.sqrt(2*pi))
            gvhT = np.piecewise(PvhT, [Temp>=Tcut(r_val),], [lambda xp:xp, lambda xp:0.])
            
            #Assumtion == Phases are internally isobaric
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
                                   
            # print('Ionization interpolation')
            fIonHot  = 10.**np.array([fIon(      nHhot[i], 
                                                 Temp[i], 
                                                 metallicity(r_val), redshift, 
                                                 element, ion, mode ) for i,xhp in enumerate(xh)])
            
            mu_hot = np.array([mu(               nHhot[i], 
                                                 Temp[i], 
                                                 metallicity(r_val), redshift, 
                                                 mode ) for i,xhp in enumerate(xh)])
            
            fIonWarm =  10.**np.array([fIon(     nHwarm[i],
                                                 Temp[i],
                                                 metallicity(r_val), redshift,
                                                 element, ion, mode) for i,xwp in enumerate(xw)])
            
            mu_warm = np.array([mu(              nHwarm[i], 
                                                 Temp[i], 
                                                 metallicity(r_val), redshift, 
                                                 mode ) for i,xhp in enumerate(xh)])

            hotInt  = (1-fvw(r_val))*np.trapz( (mu_hot*prs_hot(r_val)/(kB*Temp))*fIonHot*gvhT, xh) #global density sensitive
            
            warmInt = fvw(r_val)*np.trapz( (mu_warm*prs_warm(r_val)/(kB*Temp))*fIonWarm*gvwT, xw)
            
            nIon[indx] = a0*metallicity(r_val)*(hotInt + warmInt)*(mp/mH)*Xp(metallicity(r_val))
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(13,10))
        # ax  =  plt.gca()
        # plt.semilogy(radius, nhot_global,  label='Hot', linewidth=5)
        # plt.semilogy(radius, nwarm_global, label='Warm', linewidth=5)
        # plt.semilogy(radius, nIon, label='Ion e=%d i=%d'%(element,ion), linewidth=5)
        # ax.grid()   
        # ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
        # ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
        # ax.set_ylabel(r'Ionization fraction', size=28, color='black') 
        # ax.set_xlabel(r'Temperature [K]', size=28, color='black')
        # lgnd = ax.legend(loc='best', framealpha=0.3, prop={'size': 20}, title_fontsize=24) #, , , bbox_to_anchor=(0.88, 0))
        # ax.set_title(f'Element=%d Ion=%d Mode=%s'%(element, ion, mode), fontsize=26)
        # # plt.xlim(xmin=10**4.4, xmax=10**7.6)
        # # plt.ylim(ymin=1e-8, ymax=1.1)
        # plt.show()
        
        nIon = interp1d(radius, nIon, fill_value="extrapolate") #CGS
        # print('Ion calculation complete!')
        coldens = np.zeros_like(b_)
        
        epsilon = 1e-6
        for indx, b_val in enumerate(b_):
            coldens[indx] = 2*integrate.quad(lambda r: nIon(r)*r/np.sqrt(r**2-b_val**2), 
                                    b_val*(1+epsilon), self.redisProf.unmodified.rCGM*self.redisProf.unmodified.UNIT_LENGTH/kpc)[0] #kpc cm-3
        
        if len(b_) == 1: return coldens[0]*kpc
        else: return coldens*kpc
    
