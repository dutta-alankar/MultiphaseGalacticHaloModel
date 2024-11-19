# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:03:50 2023

@author: Alankar
"""

import numpy as np
import requests
import h5py
import io
import sys

baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key": "a85c9b695968416dfacd26146061f7d4"}


def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers["content-type"] == "application/json":
        return r.json()  # parse json responses automatically
    return r


XH = 0.76
gamma = 5 / 3.0
kB = 1.3807e-16
mp = 1.6726e-24
mH = 1.6735575e-24
pc = 3.086e18
yr = 365 * 24 * 60**2
Msun = 1.989e33

h = 0.6774

UnitLength = 1.0e+03 * pc
UnitTime = 1.0e+09 * yr
UnitMass = 1.0e+10 * Msun

UnitVelocity = UnitLength / UnitTime
UnitEnergy = UnitMass * UnitLength**2 / UnitTime**2

snapnum = 84  # z=0.2 (0.197)
haloID = int(sys.argv[1])
simulation = "TNG50-1"

sim_snaps = len(get(f"https://www.tng-project.org/api/{simulation}/snapshots/"))
halo_url = (
    f"https://www.tng-project.org/api/TNG50-1/snapshots/{snapnum}/halos/{haloID}/"
)

# Try this in case of error: {'gas':'ElectronAbundance,InternalEnergy,Density,Masses'}
cutout_query = {
    "gas": "ElectronAbundance,InternalEnergy,Density,Masses,GFM_CoolingRate,GFM_Metallicity,StarFormationRate,Coordinates",
    "stars": "Masses,Coordinates",
}
cutout = get(f"{halo_url}/cutout.hdf5", cutout_query)
halo_info = get(f"{halo_url}/info.json")
Group_CM = np.array(halo_info["GroupCM"])
Group_Pos = np.array(halo_info["GroupPos"]) # h^-1 ckpc
Group_Vel = np.array(halo_info["GroupVel"])
Group_r200 = np.array(halo_info["Group_R_Crit200"])
Group_SFR = np.array(halo_info["GroupSFR"]) # MSun/yr
Group_M200 = np.array(halo_info["Group_M_Crit200"]) # 10^10 h^-1 MSun
Group_Mwind = np.array(halo_info["GroupWindMass"]) # 10^10 h^-1 MSun
GroupMassType = np.array(halo_info["GroupMassType"]) # 10^10 h^-1 MSun

redshift = np.array(
    get(f"https://www.tng-project.org/api/{simulation}/snapshots/")[snapnum]["redshift"]
)
a = 1 / (1 + redshift)
ckpc = 1.0e+03*pc / a
UnitDensity = (UnitMass / h) / (ckpc / h) ** 3

with h5py.File(io.BytesIO(cutout.content), "r") as hdf:
    Coordinates = np.array(hdf["PartType0/Coordinates"])
    Density = np.array(hdf["PartType0/Density"])
    InternalEnergy = np.array(hdf["PartType0/InternalEnergy"])
    ElectronAbundance = np.array(hdf["PartType0/ElectronAbundance"])
    Lambda = np.array(hdf["PartType0/GFM_CoolingRate"])
    ZbZSun = np.array(hdf["PartType0/GFM_Metallicity"])/0.0127
    Masses = np.array(hdf["PartType0/Masses"])
    sfr = np.array(hdf["PartType0/StarFormationRate"])
    stellar_coordinates = np.array(hdf["PartType4/Coordinates"])
    stellar_masses = np.array(hdf["PartType4/Masses"])
    mu = 4.0 / (1 + 3 * XH + 4 * XH * ElectronAbundance)
    Temperature = (
        (gamma - 1) * (InternalEnergy * (UnitEnergy / UnitMass)) * mu * (mp / kB)
    )
    NumberDensity = (
        (gamma - 1)
        * (Density * UnitDensity)
        * (InternalEnergy * (UnitEnergy / UnitMass))
        / (kB * Temperature)
    )
    Volume = Masses * (UnitMass / h) / (Density * UnitDensity)
    Density *= UnitDensity # convert to CGS

with h5py.File(f"halo-prop_ID={haloID}.hdf5", "w") as store:
    store.create_dataset("Group_CM", data=Group_CM)
    store.create_dataset("Group_Pos", data=Group_Pos)
    store.create_dataset("Group_Vel", data=Group_Vel)
    store.create_dataset("Group_r200", data=Group_r200)
    store.create_dataset("Group_SFR", data=Group_SFR)
    store.create_dataset("Group_M200", data=Group_M200)
    store.create_dataset("Group_Mwind", data=Group_Mwind)
    store.create_dataset("Group_Mass", data=GroupMassType)

    store.create_dataset("Coordinates", data=Coordinates)
    store.create_dataset("Redshift", data=redshift)
    store.create_dataset("Density", data=Density)
    store.create_dataset("NumberDensity", data=NumberDensity)
    store.create_dataset("Temperature", data=Temperature)
    store.create_dataset("ZbZSun", data=ZbZSun)
    store.create_dataset("Volume", data=Volume)
    store.create_dataset("Masses", data=Masses)
    store.create_dataset("mu", data=mu)
    store.create_dataset("nH", data=NumberDensity * mu * XH * (mp / mH))
    store.create_dataset("Lambda", data=Lambda)
    store.create_dataset("SFR", data=sfr)
    store.create_dataset("Stellar_coordinates", data=stellar_coordinates)
    store.create_dataset("Stellar_masses", data=stellar_masses)
