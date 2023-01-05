#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:06:18 2022

@author: alankar
"""

import numpy as np
import h5py
from itertools import product
import os
import subprocess

mp     = 1.67262192369e-24
me     = 9.1093837e-28
mH     = 1.6735e-24
warn   = False
class interpolate_ionization:
    
    def __init__(self):    
        #dummy read for parameters
        _tmp = subprocess.check_output('pwd').decode("utf-8").split('/')[1:]
        _pos = None
        for i, val in enumerate(_tmp):
            if val == 'MultiphaseGalacticHaloModel':
                _pos = i
        _tmp = os.path.join('/',*_tmp[:_pos+1], 'misc', 'cloudy-data', 'ionization')
        self.loc = _tmp #'./cloudy-data/ionization'
        
        data = h5py.File('%s/ionization.b_%06d.h5'%(self.loc,0), 'r')
        self.nH_data   = np.array(data['params/nH'])
        self.T_data   = np.array(data['params/temperature'])
        self.Z_data   = np.array(data['params/metallicity'])
        self.red_data = np.array(data['params/redshift'])
        
        self.batch_size = np.prod(np.array(data['header/batch_dim']))
        self.total_size = np.prod(np.array(data['header/total_size']))
        data.close()
    
    def interpolate(self, nH=1.2e-4, temperature=2.7e6, metallicity=0.5, redshift=0.2, element=2, ion=1, mode='PIE'):
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
        if (np.sum(nH==self.nH_data)==1): 
            i_vals = [np.where(nH==self.nH_data)[0][0], np.where(nH==self.nH_data)[0][0]]
        else:
            i_vals = [np.sum(nH>self.nH_data)-1,np.sum(nH>self.nH_data)]
        
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
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH)
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH)
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
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
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH)
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH)
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
                
            d_i = np.abs(self.nH_data[i]-nH)
            d_j = np.abs(self.T_data[j]-temperature)
            d_k = np.abs(self.Z_data[k]-metallicity)
            d_l = np.abs(self.red_data[l]-redshift)
            
            #print('Data vals: ', self.nH_data[i], self.T_data[j], self.Z_data[k], self.red_data[l] )
            #print(i, j, k, l)
            epsilon = 1e-6
            weight = np.sqrt(d_i**2 + d_j**2 + d_k**2 + d_l**2 + epsilon)
            
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
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
    
    def num_dens(self, nH=1.2e-4, temperature=2.7e6, metallicity=0.5, redshift=0.2, mode='PIE', part_type='electron'):
        #print(nH, temperature, metallicity, redshift, mode, part_type)
        file_path = os.path.realpath(__file__)
        dir_loc   = os.path.split(file_path)[:-1]
        abn_file  = os.path.join(*dir_loc,'cloudy-data', 'solar_GASS10.abn')
        
        _tmp = None
        with open(abn_file, 'r') as file:
            _tmp = file.readlines()
            
        abn = np.array([ float(element.split()[-1]) for element in _tmp[2:32] ]) #till Zinc
        
        ion_count = 0
        
        for i in range(30): #till Zn
            for j in range(i+2):
                ion_count += 1
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = metallicity*Z_solar # Z varied independent of nH and nHe
    
        #element = 1: H, 2: He, 3: Li, ... 30: Zn
        #ion = 1 : neutral, 2: +, 3: ++ .... (element+1): (++++... element times)
        if (mode!='PIE' and mode!='CIE'):
            print('Problem! Invalid mode: %s.'%mode)
            return None
        
        i_vals, j_vals, k_vals, l_vals = None, None, None, None
        if (np.sum(nH==self.nH_data)==1): 
            i_vals = [np.where(nH==self.nH_data)[0][0], np.where(nH==self.nH_data)[0][0]]
        else:
            i_vals = [np.sum(nH>self.nH_data)-1,np.sum(nH>self.nH_data)]
        
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
            
        fracIon = np.zeros((ion_count,), dtype=np.float64)
        
        inv_weight = 0
        #print(i_vals, j_vals, k_vals, l_vals)
        
        batch_ids = []
        #identify unique batches
        for i, j, k, l in product(i_vals, j_vals, k_vals, l_vals):
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH, type(nH))
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH)
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
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
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH)
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH)
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
                
            d_i = np.abs(self.nH_data[i]-nH)
            d_j = np.abs(self.T_data[j]-temperature)
            d_k = np.abs(self.Z_data[k]-metallicity)
            d_l = np.abs(self.red_data[l]-redshift)
            
            #print('Data vals: ', self.nH_data[i], self.T_data[j], self.Z_data[k], self.red_data[l] )
            #print(i, j, k, l)
            epsilon = 1e-6
            weight = np.sqrt(d_i**2 + d_j**2 + d_k**2 + d_l**2 + epsilon)
            
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
                      (i)
            batch_id  = counter//self.batch_size 
            
            for id_data in data:
                if id_data[0] == batch_id:
                    hdf = id_data[1]
                    local_pos = counter%self.batch_size - 1
                    fracIon += ((np.array(hdf['output/fracIon/%s'%mode])[local_pos,:]) / weight)
                
            inv_weight += 1/weight
         
        fracIon = fracIon/inv_weight
        fracIon = 10.**fracIon
        
        for id_data in data:
            id_data[1].close()
        
        if (part_type=='all'):
            ndens = 0
            ion_count = 0
            for element in range (30):
                for ion in range (element+2):
                    if (element+1==1): #H
                        ndens += (ion+1)*(Xp/X_solar)*abn[element]*fracIon[ion_count]*nH
                    elif (element+1==2): #He
                        ndens += (ion+1)*(Yp/Y_solar)*abn[element]*fracIon[ion_count]*nH
                    else:
                        ndens += (ion+1)*(Zp/Z_solar)*abn[element]*fracIon[ion_count]*nH
                    ion_count += 1
            return ndens
        
        elif (part_type=='electron'):
            ne = 0
            ion_count = 0
            for element in range (30):
                for ion in range (element+2):
                    if (element+1==1): #H
                        ne += ion*(Xp/X_solar)*nH*abn[element]*fracIon[ion_count]
                    elif (element+1==2): #He
                        ne += ion*(Yp/Y_solar)*nH*abn[element]*fracIon[ion_count]
                    else:
                        ne += ion*(Zp/Z_solar)*nH*abn[element]*fracIon[ion_count]
                    ion_count += 1
            return ne
        
        elif (part_type=='ion'):
            nion = 0
            ion_count = 0
            for element in range (30):
                for ion in range (1,element+2):
                    if (element+1==1): #H
                        nion += (Xp/X_solar)*nH*abn[element]*fracIon[ion_count]
                    elif (element+1==2): #He
                        nion += (Yp/Y_solar)*nH*abn[element]*fracIon[ion_count]
                    else:
                        nion += ion*(Zp/Z_solar)*nH*abn[element]*fracIon[ion_count]
                    ion_count += 1
            return nion
        
        else:
            print (f'Invalid part_type: {part_type}')      
            
    def mu(self, nH=1.2e-4, temperature=2.7e6, metallicity=0.5, redshift=0.2, mode='PIE'):
        file_path = os.path.realpath(__file__)
        dir_loc   = os.path.split(file_path)[:-1]
        abn_file  = os.path.join(*dir_loc,'cloudy-data', 'solar_GASS10.abn')
        
        _tmp = None
        with open(abn_file, 'r') as file:
            _tmp = file.readlines()
        abn = np.array([ float(element.split()[-1]) for element in _tmp[2:32] ]) #till Zinc
        
        ion_count = 0
        
        for i in range(30): #till Zn
            for j in range(i+2):
                ion_count += 1
        
        X_solar = 0.7154
        Y_solar = 0.2703
        Z_solar = 0.0143
    
        Xp = X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Yp = Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
        Zp = metallicity*Z_solar # Z varied independent of nH and nHe
    
        #element = 1: H, 2: He, 3: Li, ... 30: Zn
        #ion = 1 : neutral, 2: +, 3: ++ .... (element+1): (++++... element times)
        if (mode!='PIE' and mode!='CIE'):
            print('Problem! Invalid mode: %s.'%mode)
            return None
        
        i_vals, j_vals, k_vals, l_vals = None, None, None, None
        if (np.sum(nH==self.nH_data)==1): 
            i_vals = [np.where(nH==self.nH_data)[0][0], np.where(nH==self.nH_data)[0][0]]
        else:
            i_vals = [np.sum(nH>self.nH_data)-1,np.sum(nH>self.nH_data)]
        
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
            
        fracIon = np.zeros((ion_count,), dtype=np.float64)
        
        inv_weight = 0
        #print(i_vals, j_vals, k_vals, l_vals)
        
        batch_ids = []
        #identify unique batches
        for i, j, k, l in product(i_vals, j_vals, k_vals, l_vals):
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH, type(nH))
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH, type(nH))
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
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
            if (i==self.nH_data.shape[0]): 
                if (warn): print("Problem: nH", nH)
                i = i-1
            if (i==-1): 
                if (warn): print("Problem: nH", nH)
                i = i+1
            if (j==self.T_data.shape[0]): 
                if (warn): print("Problem: T", temperature)
                j = j-1
            if (j==-1): 
                if (warn): print("Problem: T", temperature)
                j = j+1
            if (k==self.Z_data.shape[0]): 
                if (warn): print("Problem: met", metallicity)
                k = k-1
            if (k==-1): 
                if (warn): print("Problem: met", metallicity)
                k = k+1
            if (l==self.red_data.shape[0]): 
                if (warn): print("Problem: red", redshift)
                l = l-1
            if (l==-1): 
                if (warn): print("Problem: red", redshift)
                l = l+1
            d_i = np.abs(self.nH_data[i]-nH)
            d_j = np.abs(self.T_data[j]-temperature)
            d_k = np.abs(self.Z_data[k]-metallicity)
            d_l = np.abs(self.red_data[l]-redshift)
            
            #print('Data vals: ', self.nH_data[i], self.T_data[j], self.Z_data[k], self.red_data[l] )
            #print(i, j, k, l)
            epsilon = 1e-6
            weight = np.sqrt(d_i**2 + d_j**2 + d_k**2 + d_l**2 + epsilon)
            
            counter = (l)*self.Z_data.shape[0]*self.T_data.shape[0]*self.nH_data.shape[0]+ \
                      (k)*self.T_data.shape[0]*self.nH_data.shape[0] + \
                      (j)*self.nH_data.shape[0] + \
                      (i)
            batch_id  = counter//self.batch_size 
            
            for id_data in data:
                if id_data[0] == batch_id:
                    hdf = id_data[1]
                    local_pos = counter%self.batch_size - 1
                    fracIon += ((np.array(hdf['output/fracIon/%s'%mode])[local_pos,:]) / weight)
                
            inv_weight += 1/weight
         
        fracIon = fracIon/inv_weight
        fracIon = 10.**fracIon
        
        for id_data in data:
            id_data[1].close()
            
        ndens = 0
        ion_count = 0
        for element in range (30):
            for ion in range (element+2):
                if (element+1==1): #H
                    ndens += (ion+1)*(Xp/X_solar)*abn[element]*fracIon[ion_count]*nH
                elif (element+1==2): #He
                    ndens += (ion+1)*(Yp/Y_solar)*abn[element]*fracIon[ion_count]*nH
                else:
                    ndens += (ion+1)*(Zp/Z_solar)*abn[element]*fracIon[ion_count]*nH
                ion_count += 1
        # print("abn ", abn)
        # print("fracIon ", fracIon)
        # print("ndens ", ndens)        
        return (nH/ndens)*(mH/mp)/Xp
