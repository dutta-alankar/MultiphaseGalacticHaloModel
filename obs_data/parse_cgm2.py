# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:20:42 2024

@author: Alankar
"""
import os
import numpy as np

def parse_col_dens_cgm2():
    filename =  f"{os.path.dirname(__file__)}/apjac450ct3_mrt.txt"
    impact, col_dens, col_dens_err = [], [], []
    with open(filename, "r") as txt_file:
        content = txt_file.readlines()
        for indx, line in enumerate(content):
            if indx < 28:
                continue
            impact.append( float(line[63:66])/float(line[75:78]) )
            col_dens.append( float(line[79:84]) )
            col_dens_err.append( -1. if line[85] == '<' else 0. )
    return ( np.array(impact), np.array(col_dens), np.array(col_dens_err) )
