# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:20:42 2024

@author: Alankar
"""
import os
import numpy as np

def parse_col_dens_magiicat():
    filename =  f"{os.path.dirname(__file__)}/Magiicat_Paper4Table1.txt"
    impact, col_dens, col_dens_err = [], [], []
    with open(filename, "r") as txt_file:
        content = txt_file.readlines()
        for indx, line in enumerate(content):
            if indx < 2:
                continue
            line_content = line.split()
            impact.append( float(line_content[5]) )
            col_dens.append( float(line_content[15]) )
            col_dens_err.append( float(line_content[16]) )
    return ( np.array(impact), np.array(col_dens), np.array(col_dens_err) )
