# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:20:42 2024

@author: Alankar
"""

import numpy as np
import os
import sys
from typing import Optional, List
from astropy.io import fits
# from astropy.table import Table

def parse_col_dens_cubs(ion: str) -> Optional[List[np.ndarray]] :
    fits_file = fits.open(f"{os.path.dirname(__file__)}/CUBSVI_Table1.fits")
    if ion == "OVI" or ion == "NeVIII" or ion == "SVI":
        fits_file.close()
        fits_file = fits.open(f"{os.path.dirname(__file__)}/CUBSVII_tableB1.fits")
    # fits_file.info()

    # print(fits_file[1].data)
    table = fits_file[1].data

    qso, redshift, impacts = [], [], []
    b_vals, col_dens_obs, col_dens_err_obs = [], [], []

    for observe in table:
        if ion == "OVI" or ion == "NeVIII" or ion == "SVI":
            if ion == "OVI":
                ion_pos = 30
            elif ion == "NeVIII":
                ion_pos = 39
            elif ion == "SVI":
                ion_pos = 48
            b_vals.append(observe[5]/observe[9])
            col_dens_obs.append(observe[ion_pos])
            col_dens_err_obs.append([observe[ion_pos+1], observe[ion_pos+2]])
        else:
            qso.append(observe[1])
            redshift.append(observe[2])
            impacts.append(observe[15])

    fits_file.close()
    if ion == "OVI" or ion == "NeVIII" or ion == "SVI":
        return (np.array(b_vals), np.array(col_dens_obs), np.array(col_dens_err_obs).T)

    fits_file = fits.open(f"{os.path.dirname(__file__)}/CUBSz1_detected.fits")
    # fits_file.info()

    table = fits_file[1].data

    ions = []
    for observe in table:
        ions.append(observe[2])
    ions = set(ions)

    if ion not in ions:
        print(f"{ion} absent in CUBS VI/VII data!")
        sys.exit(1)

    for indx, impact in enumerate(impacts):
        col_dens_tot = 0
        col_dens_err_tot = [0, 0]
        for observe in table:
            abs_qso = observe[0]
            abs_redshift = observe[1]
            abs_ion = observe[2]
            col_dens = observe[3]
            col_dens_err = [observe[4], observe[5]]
            if abs_ion != ion:
                continue
            if (abs_redshift == redshift[indx]) and (abs_qso == qso[indx]):
                if (col_dens_err[0] != 1.) and (col_dens_err[0] != -1.):
                    col_dens_tot = np.log10( np.power(10., col_dens_tot) + np.power(10., col_dens) )
                    col_dens_err_tot[0] = np.sqrt(col_dens_err_tot[0]**2 + col_dens_err[0]**2)
                    col_dens_err_tot[1] = np.sqrt(col_dens_err_tot[1]**2 + col_dens_err[1]**2)
                else:
                    col_dens_tot = np.sqrt(col_dens_tot**2 + col_dens**2)
                    col_dens_err_tot[0] = col_dens_err[0]
                    col_dens_err_tot[1] = col_dens_err[1]
        if col_dens_tot != 0:
            b_vals.append(impact)
            col_dens_obs.append(col_dens_tot)
            col_dens_err_obs.append(col_dens_err_tot)
    fits_file.close()

    return (np.array(b_vals), np.array(col_dens_obs), np.array(col_dens_err_obs).T)

# print(parse_col_dens_cubs("OVI")[2][:,17])
# print(parse_col_dens_cubs("OVI")[1][17])