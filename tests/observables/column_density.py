#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:25:15 2023

@author: alankar
"""

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../submodules/AstroPlasma')
import numpy as np
import pickle
from itertools import product
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isobarcool import IsobarCoolRedistribution
from modified.isochorcool import IsochorCoolRedistribution
from observable.ColumnIon import ion_column

def profile_gen(unmod: str, mod: str, ionization: str) -> None: 
    print(unmod, mod, ionization)
    
    cutoff = 4.0
    TmedVW=3.e5
    sig = 0.3
    redshift = 0.001
    
    if unmod=="isoth":
        TmedVH=1.5e6
        THotM = TmedVH*np.exp(-sig**2/2)
        unmodified = IsothermalUnmodified(THot=THotM,
                              P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                              M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                              redshift=redshift, ionization=ionization)
    else:
        nHrCGM = 1.1e-5
        TthrCGM = 2.4e5
        sigmaTurb = 60
        ZrCGM = 0.3
        unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                          redshift=redshift, ionization=ionization)
        
    if mod=="isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff, isobaric=0)
        
    column = ion_column(modified)
    
    ions = ["OVI", "OVII", "OVIII", "NV"]
    
    print("Evaluating Column densities...")
    
    pc     = 3.0856775807e18
    kpc    = 1e3*pc
    
    b = np.linspace(9.0, unmodified.rCGM*unmodified.UNIT_LENGTH/kpc, 200) # kpc
    
    for ion in ions:
        print(ion)   
        column_density = column.gen_column(b, element=ion)
    
        with open(f'figures/N_{ion}_{unmod}_{mod}_{ionization}.pickle', 'wb') as f:
            data = {'impact': b, 
                    f'N_{ion}': column_density,
                    'rCGM': unmodified.rCGM*unmodified.UNIT_LENGTH/kpc,}
            pickle.dump(data, f)
    

if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]
    
    for condition in product(unmod, mod, ionization):
        profile_gen(*condition)
    