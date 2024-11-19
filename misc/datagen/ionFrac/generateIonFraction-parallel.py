#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:47:02 2022

@author: alankar
"""

import time
import numpy as np
import os
from itertools import product
import matplotlib.pyplot as plt
import sys
import h5py
from ProgressBar import ProgressBar as pbar

from mpi4py import MPI

## start parallel programming ---------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
startTime =  time.time()

resume = False
start  = 249800
cloudy_done = False

def ionFrac(ndens=1e-4, temperature=1e6, metallicity=1.0, redshift=0., indx=99999):
    plotIt = False
    
    pi     = np.pi
    pc     = 3.0856775807e18
    kpc    = 1e3*pc
    Mpc    = 1e3*kpc
    e      = 4.8032e-10
    s      = 1
    cm     = 1
    K      = 1
    km     = 1e5*cm
    mp     = 1.67262192369e-24
    kB     = 1.3806505e-16
    G      = 6.6726e-8
    H0     = 67.4
    H0cgs  = H0*((km/s)/Mpc)
    dcrit0 = 3*H0cgs**2/(8.*pi*G) 
    mu     = 0.6
    Msun   = 2.e33
    
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
    
    os.system(f"./ionization_CIE {nH:.2e} {temperature:.2e} {metallicity:.2e} {redshift:.2f} {indx} > ./auto/auto_{indx:09}.out")     
    os.system(f"./ionization_PIE {nH:.2e} {temperature:.2e} {metallicity:.2e} {redshift:.2f} {indx} > ./auto/auto_{indx:09}.out")
    os.system(f"rm -rf ./auto/auto_{indx:09}.out")
    return None

if not(cloudy_done):
    if rank==0 and not(resume):
        if (os.path.exists('./auto')):
            os.system('rm -rf ./auto')
        os.system('mkdir -p ./auto')
    sys.stdout.flush()
    comm.Barrier()
    
    total_size = [50,50,20,5]
    
    ndens       = np.logspace(-6, 0, total_size[0])
    temperature = np.logspace(3.8, 8, total_size[1])
    metallicity = np.logspace(-1, 1, total_size[2])
    redshift    = np.linspace(0, 2, total_size[3])
    
    batch_dim  = [8,8,4,2]
    batches    = int(np.prod(total_size)//np.prod(batch_dim)) + (0 if (np.prod(total_size)%np.prod(batch_dim))==0 else 1)
    
    values  = list(product(redshift, metallicity, temperature, ndens )) #itertools are lazy
    offsets = np.hstack((np.arange(0, np.prod(total_size), np.prod(batch_dim)), [len(values),]))
    sys.stdout.flush()
    comm.Barrier()
    
    if (resume and rank==0): print("Resuming from stopped state ... ")
    
    progbar = None       
    if rank == 0: progbar = pbar()   
    sys.stdout.flush()
    
    start = start - size            
    for indx in range(rank, len(values), size):
        this_val = values[indx]    
            
        i, j, k, l = np.where(ndens==this_val[-1])[0][0],\
                     np.where(temperature==this_val[-2])[0][0],\
                     np.where(metallicity==this_val[-3])[0][0], \
                     np.where(redshift==this_val[-4])[0][0]
        counter    = (l)*metallicity.shape[0]*temperature.shape[0]*ndens.shape[0]+ \
                     (k)*temperature.shape[0]*ndens.shape[0] + \
                     (j)*ndens.shape[0] + \
                     (i)
        if (resume and (counter<=start)): 
            if rank==0: progbar.progress(min(indx+size, len(values)), len(values))
            continue
        else:
            ionFrac(*this_val[::-1], counter)
        sys.stdout.flush()
        if rank==0: progbar.progress(min(indx+size, len(values)), len(values))
        sys.stdout.flush()
        
    if rank==0 : progbar.end()
    sys.stdout.flush()
    comm.Barrier()

progbar = None       
if rank == 0: 
    print("Collecting into batches ... ")
    progbar = pbar()
sys.stdout.flush()

N_max = 30     #element Zinc
for batch_id in range(rank,batches,size):    
    this_batch_size = offsets[batch_id+1]-offsets[batch_id]
    fracCIE = np.zeros( (this_batch_size,int(N_max*(N_max+3)/2)) )
    fracPIE = np.zeros_like(fracCIE)
    count = 0
    for this_val in values:       
        i, j, k, l = np.where(ndens==this_val[-1])[0][0],\
                     np.where(temperature==this_val[-2])[0][0],\
                     np.where(metallicity==this_val[-3])[0][0], \
                     np.where(redshift==this_val[-4])[0][0]
        counter    = (l)*metallicity.shape[0]*temperature.shape[0]*ndens.shape[0]+ \
                     (k)*temperature.shape[0]*ndens.shape[0] + \
                     (j)*ndens.shape[0] + \
                     (i)
        in_this_batch = counter>=offsets[batch_id] and counter<offsets[batch_id+1]
        if not( in_this_batch ):
            continue
        fracCIE[count,:] = np.loadtxt("./auto/ionization_CIE_%09d.txt"%counter, dtype=np.float32)
        fracPIE[count,:] = np.loadtxt("./auto/ionization_PIE_%09d.txt"%counter, dtype=np.float32)
        count += 1
        
    
    #np.save('ionization.npy', data, allow_pickle=True)
    loc = '.'
    if not(os.path.exists('%s/data'%loc)): 
        os.system('mkdir -p %s/data'%loc)
    with h5py.File('%s/data/ionization.b_%06d.h5'%(loc,batch_id), 'w') as hdf:
        hdf.create_dataset('params/ndens', data=ndens)
        hdf.create_dataset('params/temperature', data=temperature)
        hdf.create_dataset('params/metallicity', data=metallicity)
        hdf.create_dataset('params/redshift', data=redshift)
        hdf.create_dataset('output/fracIon/CIE', data=fracCIE)
        hdf.create_dataset('output/fracIon/PIE', data=fracPIE)  
        hdf.create_dataset('header/batch_id', data=batch_id)
        hdf.create_dataset('header/batch_dim', data=batch_dim)
        hdf.create_dataset('header/total_size', data=total_size)
        
    if (rank==0): progbar.progress(min(batch_id+size, batches), batches) 
    sys.stdout.flush()
comm.Barrier()
if (rank==0): progbar.end()
sys.stdout.flush()
stopTime  = time.time()
    
if rank==0: 
    print("Elapsed: %.2f s"%(stopTime-startTime))
sys.stdout.flush()
comm.Disconnect()

