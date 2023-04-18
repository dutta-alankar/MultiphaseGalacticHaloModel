#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:15:04 2023

@author: alankar
"""

import sys
import os
sys.path.append('..')
import numpy as np
from scipy import interpolate
from misc.constants import kpc, mH
from unmodified.isoth import IsothermalUnmodified
# from unmodified.isent import IsentropicUnmodified
# from observable.ColumnDensity import ColumnDensityGen
from misc.ionization import interpolate_ionization
import matplotlib.pyplot as plt

np.random.seed(10)

element=12
ion=2

file_path = os.path.realpath(__file__)
dir_loc   = os.path.split(file_path)[:-1]
abn_file  = os.path.join(*dir_loc,'..','misc','cloudy-data', 'solar_GASS10.abn')
_tmp = None
with open(abn_file, 'r') as file:
    _tmp = file.readlines()
    
abn = np.array([ float(element.split()[-1]) for element in _tmp[2:32] ]) #till Zinc
a0  = abn[element-1]
fIon = interpolate_ionization().fIon
mu = interpolate_ionization().mu

X_solar = 0.7154
Y_solar = 0.2703
Z_solar = 0.0143

Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Zp = lambda metallicity: metallicity*Z_solar


# -----------------------------------------
TmedVH = pow(10., 5.636)
TmedVW = pow(10., 4.101)
sig = 0.3
THotM = TmedVH*np.exp(-sig**2/2)

radius = np.linspace(9.0,250,30) # kpc

# PIE
unmodified = IsothermalUnmodified(THot=THotM,
                      P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                      M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                      redshift=0.001, ionization='PIE')
rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu_val = unmodified.ProfileGen(radius)


prs = interpolate.interp1d(radius, 
                           prsTh, 
                           fill_value='extrapolate') # CGS
density = interpolate.interp1d(radius, 
                           rho, 
                           fill_value='extrapolate') # CGS
metallicity = interpolate.interp1d(radius, 
                                   unmodified.metallicity, 
                                   fill_value='extrapolate')

print('Hot phase calculation complete!')

def intersect_clouds(clouds, cloud_size, los, rCGM):
    b = los['b']
    phi = los['phi']
    
    los_x = b*np.cos(phi)
    los_y = b*np.sin(phi)
    
    r1 = np.array([los_x, los_y, 0.])
    r2 = np.array([los_x, los_y, np.sqrt(rCGM**2-b**2)])
    
    cloud_intersect = []
    for num, cloud in enumerate(clouds):
        clpos_x = cloud['x']
        clpos_y = cloud['y']
        clpos_z = cloud['z']
        point = np.array([clpos_x, clpos_y, clpos_z])
        
        distance = np.linalg.norm(np.cross( (point-r1), (r2-r1) ))/np.linalg.norm(r2-r1)
        if (distance<cloud_size): 
            nIonCloud = a0*cloud['met']*cloud['fIon']*Xp(cloud['met'])*cloud['rho']/(cloud['mu']*mH)
            cloud_intersect.append([num, 2*np.sqrt(cloud_size**2-distance**2), nIonCloud])
    return  cloud_intersect

r0 = unmodified.Halo.r0*unmodified.Halo.UNIT_LENGTH/kpc #kpc
rCGM = 1.1*unmodified.Halo.r200*unmodified.Halo.UNIT_LENGTH/kpc #kpc

fvw = 1 - (0.089 + 0.907)
n_wcl = int(1e3)

rwarm = rCGM*(fvw/n_wcl)**(1./3)
print(f'Cloud size {rwarm*1e3:.1f} pc')

pos_warm = [np.random.uniform(r0+rwarm, rCGM-rwarm, n_wcl),
            np.random.uniform(0., np.pi, n_wcl),
            np.random.uniform(0., 2*np.pi, n_wcl),]
pos_warm = np.array([pos_warm[0]*np.sin(pos_warm[1])*np.cos(pos_warm[2]),
                     pos_warm[0]*np.sin(pos_warm[1])*np.sin(pos_warm[2]),
                     pos_warm[0]*np.cos(pos_warm[1])]).T

distance_cloud = np.sqrt(np.sum(pos_warm**2, axis=1))
met_cloud = metallicity(distance_cloud)
rho_cloud = density(distance_cloud)*(TmedVH/TmedVW) # press-eq

clouds = [{'x': pos_warm[i,0], 'y': pos_warm[i,1], 'z': pos_warm[i,2],
           'dist': distance_cloud[i],
           'Temp': TmedVW, 
           'prs': prs(distance_cloud[i]), 
           'met': met_cloud[i],
           'rho': rho_cloud[i],
           'fIon': pow(10.,fIon(nH=rho_cloud[i]*Xp(met_cloud[i])/mH,
                        temperature=TmedVW, 
                        metallicity=met_cloud[i], 
                        redshift=0.001,
                        element=element, ion=ion,
                        mode='PIE')),
           'mu': mu(nH=rho_cloud[i]*Xp(met_cloud[i])/mH,
                        temperature=TmedVW, 
                        metallicity=met_cloud[i], 
                        redshift=0.001,
                        mode='PIE'),
           } for i in range(n_wcl)]
print('Cloud populating complete!')

phi = np.linspace(0, 2*np.pi, 10)
impact = np.linspace(r0, rCGM-r0, 20)

col_dens = np.zeros((impact.shape[0], phi.shape[0]), dtype=np.float32)

for i in range(impact.shape[0]):
    for j in range(phi.shape[0]):
        los = {'b': impact[i], 
               'phi': phi[j],}
        tot_proj_length = 2*np.sqrt(rCGM**2-impact[i]**2)
        cloud_intersect = intersect_clouds(clouds, rwarm, los, rCGM)
        if len(cloud_intersect)>0:
            col_dens[i,j] = np.sum(np.array(cloud_intersect)[:,1]*np.array(cloud_intersect)[:,2])
        else:
            col_dens[i,j] = 0. #nhot*tot_proj_length
col_dens *= kpc
# each row corresponds to same impact parameter

save_dic = {'impact': impact,
            'col_dens': col_dens,
            'rCGM': rCGM}

np.savez(f'randomSight_e.{element}_i.{ion}.npz', 
         impact=impact, 
         col_dens=col_dens,
         rCGM=rCGM,)

for i in range(impact.shape[0]):
    plt.scatter((impact[i]/rCGM)*np.ones(col_dens.shape[1]), col_dens[i,:], color='tab:blue')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlabel('b/r200')
plt.ylabel(r'Column Density [$cm^{-2}$]')
plt.show()
