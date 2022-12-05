#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:10:59 2022

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ionization import interpolate_ionization

ionFrac = interpolate_ionization()
ndens = 1.1e-4
temperature = np.logspace(4,6,20)
metallicity = 1.0
redshift = 0.1

#OVI
element = 8
ion = 6

start_time = time.time()
fracIon_PIE, fracIon_CIE = np.zeros_like(temperature), np.zeros_like(temperature)
for i in range(temperature.shape[0]):
    fracIon_PIE[i] = 10**ionFrac.interpolate(ndens, temperature[i], metallicity, redshift, 
                                             element=element, ion=ion, mode='PIE')
    fracIon_CIE[i] = 10**ionFrac.interpolate(ndens, temperature[i], metallicity, redshift, 
                                             element=element, ion=ion, mode='CIE')
stop_time = time.time()
print('Elapsed %.2f s in interpolation'%(stop_time-start_time))
    
fig = plt.figure(figsize=(13,10))
ax  =  plt.gca()
plt.loglog(temperature, fracIon_PIE, label='interpolated PIE', linewidth=5)
plt.loglog(temperature, fracIon_CIE, label='interpolated CIE', linewidth=5)
ax.grid()   
ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
ax.set_ylabel(r'Ionization fraction', size=28, color='black') 
ax.set_xlabel(r'Temperature [K]', size=28, color='black')
plt.xlim(xmin=9.9e3, xmax=1.1e6)
plt.ylim(ymin=1e-8, ymax=1.1)

X_solar = 0.7154
Y_solar = 0.2703
Z_solar = 0.0143

Xp = X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Yp = Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Zp = metallicity*Z_solar # completely ionized plasma; Z varied independent of nH and nHe; Metals==Oxygen

mu   = 1./(2*Xp+0.75*Yp+0.5625*Zp)
mup  = 1./(2*Xp+0.75*Yp+(9./16.)*Zp)
muHp = 1./Xp
mue  = 2./(1+Xp)
mui  = 1./(1/mu-1/mue)
  
nH    = (mu/muHp)*ndens # cm^-3

if (os.path.exists('./auto')):
    os.system(f"rm -rf ./auto")
os.system('mkdir -p ./auto') 
    
start_time = time.time()
for i in range(temperature.shape[0]):
    indx=3000000
    slice_start, slice_stop = int((element-1)*(element+2)/2), int(element*(element+3)/2)
    
    os.system(f"./ionization_CIE {nH:.2e} {temperature[i]:.2e} {metallicity:.2e} {redshift:.2f} {indx} > ./auto/auto_{indx:09}.out")  
    fracIon_CIE[i] = 10**(np.loadtxt("./auto/ionization_CIE_%09d.txt"%indx)[slice_start:slice_stop])[ion-1]
     
    os.system(f"./ionization_PIE {nH:.2e} {temperature[i]:.2e} {metallicity:.2e} {redshift:.2f} {indx} > ./auto/auto_{indx:09}.out")
    fracIon_PIE[i] = 10**(np.loadtxt("./auto/ionization_PIE_%09d.txt"%indx)[slice_start:slice_stop])[ion-1]  
    
    os.system(f"rm -rf ./auto/auto_{indx:09}.out")
    
stop_time = time.time()
os.system(f"rm -rf ./auto")
print('Elapsed %.2f s in cloudy runs'%(stop_time-start_time))

plt.loglog(temperature, fracIon_PIE, label='cloudy PIE', linewidth=5)
plt.loglog(temperature, fracIon_CIE, label='cloudy CIE', linewidth=5)

lgnd = ax.legend(loc='lower left', framealpha=0.3, prop={'size': 20}, title_fontsize=24) #, , , bbox_to_anchor=(0.88, 0))
ax.set_title(f'ndens={ndens:.2e} {r"$cm^{-3}$"}, Z={metallicity:.2f} Zsolar, z={redshift:.2f}', fontsize=26)
plt.show()