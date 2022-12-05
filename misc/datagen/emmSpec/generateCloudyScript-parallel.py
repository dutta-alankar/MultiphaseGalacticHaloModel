#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:34:06 2022

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
start  = 249900

def emmisivity(ndens=1e-4, temperature=1e6, metallicity=1.0, redshift=0., indx=None, mode='PIE'):
    plotIt = False
    
    if (mode!='CIE') and (mode!='PIE'):
        if (rank==0): print('Problem!')
        comm.Disconnect()
        sys.exit(1)
    
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

    background = f"table HM12 redshift {redshift:.2f}" if mode=='PIE' else '\n'
    stream = \
    f"""
# ----------- Auto generated from generateCloudyScript.py ----------------- 
#Created on {time.ctime()}
#
#@author: alankar
#
CMB redshift {redshift:.2f}
{background}
sphere
radius 150 to 151 linear kiloparsec
##table ISM
abundances "solar_GASS10.abn"
metals {metallicity:.2e} linear
hden {np.log10(nH):.2f} log
constant temperature, T={temperature:.2e} K linear
stop zone 1
iterate convergence
age 1e9 years
##save continuum "spectra.lin" units keV no isotropic
save diffuse continuum "{"emission_%s.lin"%mode if indx==None else "emission_%s_%09d.lin"%(mode,indx)}" units keV 
    """
    stream = stream[1:]
    
    if rank==0:
        if not(os.path.isfile('./auto')): 
            os.system('mkdir -p ./auto')
        if not(os.path.isfile('./auto/cloudy.exe')): 
            os.system('cp ./cloudy.exe ./auto')
        if not(os.path.isfile('./auto/libcloudy.so')): 
            os.system('cp ./libcloudy.so ./auto')
    comm.Barrier()    
    if indx!=None: filename = "autoGenScript_%s_%09d.in"%(mode,indx)
    else: filename = "autoGenScript.in"
    with open("./auto/%s"%filename, "w") as text_file:
        text_file.write(stream)
        
    os.system("cd ./auto && ./cloudy.exe -r %s"%filename[:-3])
    data = np.loadtxt("./auto/emission_%s.lin"%mode if indx==None else "./auto/emission_%s_%09d.lin"%(mode,indx) )
    if indx==None: 
        os.system('rm -rf ./auto/emission_%s.lin'%mode)
        os.system('rm -rf ./auto/%s'%filename)
    os.system('rm -rf ./auto/%s.out'%filename[:-3])
        
    if plotIt:
        delV = 4*np.pi*(150*kpc)**2*(151-150)*kpc
        fig = plt.figure(figsize=(13,10))
        ax  =  plt.gca()
        plt.loglog(data[:,0], data[:,-1]/data[:,0]*delV/(4*np.pi))
        ax.grid()   
        ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
        ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
        ax.set_ylabel(r'erg/s/keV/sr', size=28, color='black') 
        ax.set_xlabel(r'E (keV)', size=28, color='black')
        ax.set_xlim(xmin=5e-3, xmax=2e1)
        ax.set_ylim(ymin=1e29, ymax=1e44)
        #lgnd = ax.legend(loc='lower left', framealpha=0.3, prop={'size': 20}, title_fontsize=24) #, , , bbox_to_anchor=(0.88, 0))
        #lgnd.set_title(r'$\rm R_{cl}$')
        fig.tight_layout()
        #plt.savefig('./plots-res-rho.png', transparent=True, bbox_inches='tight')
        plt.show()
        plt.close(fig)   
    
    return np.vstack((data[:,0],data[:,-1])).T

if rank==0 and not(resume):
    if (os.path.exists('./auto')):
        os.system('rm -rf ./auto')
comm.Barrier()

total_size = [50,50,20,5]

ndens       = np.logspace(-6, 0, total_size[0])
temperature = np.logspace(3.8, 8, total_size[1])
metallicity = np.logspace(-1, 1, total_size[2])
redshift    = np.linspace(0, 2, total_size[3])

batch_dim  = [8,8,4,2]
batches    = int(np.prod(total_size)//np.prod(batch_dim)) + (0 if (np.prod(total_size)%np.prod(batch_dim))==0 else 1)

values  = list(product(redshift, metallicity, temperature, ndens )) #itertools are lazy
sample  = emmisivity(indx=rank if not(resume) else (len(values)+10*rank))
energy  = sample[:,0]
offsets = np.hstack((np.arange(0, np.prod(total_size), np.prod(batch_dim)), [len(values),]))
comm.Barrier()

if rank==0 and not(resume): os.system('rm -rf ./auto')
if (resume and rank==0): print("Resuming from stopped state ... ")

progbar = None       
if rank == 0: progbar = pbar()   

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
        spectrum = emmisivity(*this_val[::-1], counter, 'CIE')
        spectrum = emmisivity(*this_val[::-1], counter, 'PIE')
    
    if rank==0: progbar.progress(min(indx+size, len(values)), len(values))
    
if rank==0 : progbar.end()
comm.Barrier()

progbar = None       
if rank == 0: 
    print("Collecting into batches ... ")
    progbar = pbar()
    
for batch_id in range(rank,batches,size):    
    this_batch_size = offsets[batch_id+1]-offsets[batch_id]
    tot_emm_CIE  = np.zeros((this_batch_size, sample.shape[0])) 
    cont_emm_CIE = np.zeros_like(tot_emm)
    tot_emm_PIE  = np.zeros((this_batch_size, sample.shape[0])) 
    cont_emm_PIE = np.zeros_like(tot_emm)
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
        spectrum_CIE = np.loadtxt("./auto/emission_CIE_%09d.lin"%counter, dtype=np.float32)
        tot_emm_CIE[count,:]  = spectrum_CIE[:,-1]
        cont_emm_CIE[count,:] = spectrum_CIE[:,1]
        
        spectrum_PIE = np.loadtxt("./auto/emission_PIE_%09d.lin"%counter, dtype=np.float32)
        tot_emm_PIE[count,:]  = spectrum_PIE[:,-1]
        cont_emm_PIE[count,:] = spectrum_PIE[:,1]
        
        count += 1
        
    
    #np.save('emission.npy', data, allow_pickle=True)
    loc = '/run/user/1000/gvfs/sftp:host=10.42.75.223,user=alankar/home/alankar/cloudy-interp/'
    if not(os.path.isfile('%s/data'%loc)): 
        os.system('mkdir -p %s/data'%loc)
    with h5py.File('%s/data/emission.b_%06d.h5'%(loc,batch_id), 'w') as hdf:
        hdf.create_dataset('params/ndens', data=ndens)
        hdf.create_dataset('params/temperature', data=temperature)
        hdf.create_dataset('params/metallicity', data=metallicity)
        hdf.create_dataset('params/redshift', data=redshift)
        hdf.create_dataset('output/emission/CIE/continuum', data=cont_emm_CIE)
        hdf.create_dataset('output/emission/CIE/total', data=tot_emm_CIE)  
        hdf.create_dataset('output/emission/PIE/continuum', data=cont_emm_PIE)
        hdf.create_dataset('output/emission/PIE/total', data=tot_emm_PIE) 
        hdf.create_dataset('output/energy', data=energy)
        hdf.create_dataset('header/batch_id', data=batch_id)
        hdf.create_dataset('header/batch_dim', data=batch_dim)
        hdf.create_dataset('header/total_size', data=total_size)
        
    if (rank==0): progbar.progress(min(batch_id+size, batches), batches) 
comm.Barrier()
if (rank==0): progbar.end()
stopTime  = time.time()
    
if rank==0: 
    print("Elapsed: %.2f s"%(stopTime-startTime))
comm.Disconnect()
