#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:02:30 2022

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt

isoth_ic_CIE = np.load('./isoth/emm-spec-ic_CIE.npy')
isoth_ic_PIE = np.load('./isoth/emm-spec-ic_PIE.npy')
isent_ic_CIE = np.load('./isent/emm-spec-ic_CIE.npy')
isent_ic_PIE = np.load('./isent/emm-spec-ic_PIE.npy')

plt.figure(figsize=(13,10))
plt.plot(isoth_ic_PIE[:,0], isoth_ic_PIE[:,1]/(4*np.pi), color='mediumvioletred', label=r'Isothermal PIE', linewidth=1.5)
# plt.plot(isoth_ic_CIE[:,0], isoth_ic_CIE[:,1]/(4*np.pi), color='teal',  label=r'Isothermal CIE', linewidth=1.5, alpha=0.8)

plt.plot(isent_ic_PIE[:,0], isent_ic_PIE[:,1]/(4*np.pi), color='teal',  label=r'Isentropic PIE', linewidth=1.5, alpha=0.8)
# plt.plot(isent_ic_CIE[:,0], isent_ic_CIE[:,1]/(4*np.pi), color='tab:blue',  label=r'Isentropic CIE', linewidth=1.5, alpha=0.8)

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1e-3, xmax=1.5e1)
plt.ylim(ymin=1e29, ymax=1e47)
plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
plt.ylabel(r'$L_{\nu}/4\pi$ = $\int$ 4$\pi$ $j_{\nu}r^2dr$ / $4\pi$ [$erg\; s^{-1} \; keV^{-1} sr^{-1}$]', size=28, color='black') 

plt.xlabel(r'Energy [$keV$]', size=28, color='black')
leg = plt.legend(loc='lower left', ncol=1, fancybox=True, fontsize=25, framealpha=0.5)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()

plt.axvspan(0.3, 0.6, color='darkkhaki', alpha=0.4, zorder=0)
plt.axvspan(0.3, 2.0, color='lightsteelblue', alpha=0.5, zorder=-1)

# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)
    
plt.savefig('EmissionSpectrum.png', transparent=True)
plt.show()
plt.close()

'''
SB PIE isotherm (0.3-0.6 keV): 4.81e-11 erg cm^-2 s^-1 deg^-2
SB PIE isotherm (0.3-2.0 keV): 6.48e-11 erg cm^-2 s^-1 deg^-2
SB CIE isotherm (0.3-0.6 keV): 4.57e-11 erg cm^-2 s^-1 deg^-2
SB CIE isotherm (0.3-2.0 keV): 5.54e-11 erg cm^-2 s^-1 deg^-2
SB PIE isentrp (0.3-0.6 keV): 1.76e-09 erg cm^-2 s^-1 deg^-2
SB PIE isentrp (0.3-2.0 keV): 1.79e-09 erg cm^-2 s^-1 deg^-2
SB CIE isentrp (0.3-0.6 keV): 1.75e-09 erg cm^-2 s^-1 deg^-2
SB CIE isentrp (0.3-2.0 keV): 1.78e-09 erg cm^-2 s^-1 deg^-2
'''