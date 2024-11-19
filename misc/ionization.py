#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:06:18 2022

@author: alankar
"""

import numpy as np
import h5py
from itertools import product

class interpolate_ionization:
    
    def __init__(self):    
        #dummy read for parameters
        self.loc = './cloudy-data/ionization'
        data = h5py.File('%s/ionization.b_%06d.h5'%(self.loc,0), 'r')
        self.n_data   = np.array(data['params/ndens'])
        self.T_data   = np.array(data['params/temperature'])
        self.Z_data   = np.array(data['params/metallicity'])
        self.red_data = np.array(data['params/redshift'])
        
        self.batch_size = np.prod(np.array(data['header/batch_dim']))
        self.total_size = np.prod(np.array(data['header/total_size']))
        data.close()
    
    def interpolate(self, ndens=1.2e-4, temperature=2.7e6, metallicity=0.5, redshift=0.2, element=2, ion=1, mode='PIE'):
        #element = 1: H, 2: He, 3: Li, ... 30: Zn
        #ion = 1 : neutral, 2: +, 3: ++ .... (element+1): (++++... element times)
        if (mode!='PIE' and mode!='CIE'):
            print('Problem! Invalid mode: %s.'%mode)
            return None
        if (ion<0 or ion>element+1):
            print('Problem! Invalid ion %d for element %d.'%(ion,element))
            return None
        if (element<0 or element>30):
            print('Problem! Invalid element %d.'%element)
            return None
        
        i_vals, j_vals, k_vals, l_vals = None, None, None, None
        if (np.sum(ndens==self.n_data)==1): 
            i_vals = [np.where(ndens==self.n_data)[0][0], np.where(ndens==self.n_data)[0][0]]
        else:
            i_vals = [np.sum(ndens>self.n_data)-1,np.sum(ndens>self.n_data)]
        
        if (np.sum(temperature==self.T_data)==1): 
            j_vals = [np.where(temperature==self.T_data)[0][0], np.where(temperature==self.T_data)[0][0]]
        else:
            j_vals = [np.sum(temperature>self.T_data)-1,np.sum(temperature>self.T_data)]
        
        if (np.sum(metallicity==self.Z_data)==1): 
            k_vals = [np.where(metallicity==self.Z_data)[0][0], np.where(metallicity==self.Z_data)[0][0]]
        else:
            k_vals = [np.sum(metallicity>self.Z_data)-1,np.sum(metallicity>self.Z_data)]
        
        if (np.sum(redshift==self.red_data)==1): 
            l_vals = [np.where(redshift==self.red_data)[0][0], np.where(redshift==self.red_data)[0][0]]
        else:
            l_vals = [np.sum(redshift>self.red_data)-1,np.sum(redshift>self.red_data)]
            
        fracIon = np.zeros((element+1,), dtype=np.float64)
        
        inv_weight = 0
        #print(i_vals, j_vals, k_vals, l_vals)
        
        batch_ids = []
        #identify unique batches
        for i, j, k, l in product(i_vals, j_vals, k_vals, l_vals):
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.n_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.n_data.shape[0] + \
                      (j)*self.n_data.shape[0] + \
                      (i)
            batch_id  = counter//self.batch_size 
            batch_ids.append(batch_id)
        batch_ids = set(batch_ids)
        #print("Batches involved: ", batch_ids)
        data = []
        for batch_id in batch_ids:
            hdf = h5py.File('%s/ionization.b_%06d.h5'%(self.loc,batch_id), 'r')
            data.append([batch_id, hdf])
        
        for i, j, k, l in product(i_vals, j_vals, k_vals, l_vals):
            d_i = np.abs(self.n_data[i]-ndens)
            d_j = np.abs(self.T_data[j]-temperature)
            d_k = np.abs(self.Z_data[k]-metallicity)
            d_l = np.abs(self.red_data[l]-redshift)
            
            #print('Data vals: ', self.n_data[i], self.T_data[j], self.Z_data[k], self.red_data[l] )
            #print(i, j, k, l)
            epsilon = 1e-6
            weight = np.sqrt(d_i**2 + d_j**2 + d_k**2 + d_l**2 + epsilon)
            
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.n_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.n_data.shape[0] + \
                      (j)*self.n_data.shape[0] + \
                      (i)
            batch_id  = counter//self.batch_size 
            
            for id_data in data:
                if id_data[0] == batch_id:
                    hdf = id_data[1]
                    local_pos = counter%self.batch_size - 1
                    slice_start, slice_stop = int((element-1)*(element+2)/2), int(element*(element+3)/2) 
                    fracIon += ((np.array(hdf['output/fracIon/%s'%mode])[local_pos,slice_start:slice_stop]) / weight)
                
            inv_weight += 1/weight
         
        fracIon = fracIon/inv_weight
        
        for id_data in data:
            id_data[1].close()
        
        #array starts from 0 but ion from 1            
        return fracIon[ion-1] 