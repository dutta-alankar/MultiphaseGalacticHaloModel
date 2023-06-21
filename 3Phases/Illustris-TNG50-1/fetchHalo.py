# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:03:50 2023

@author: Alankar
"""

import numpy as np
import requests
import h5py
import io

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

UnitLength = 1e3 * pc
UnitTime = 1e9 * yr
UnitMass = 1e10 * Msun


UnitVelocity = UnitLength / UnitTime
UnitEnergy = UnitMass * UnitLength**2 / UnitTime**2

snapnum = 84  # z=0.2
haloID = 110
simulation = "TNG50-1"

sim_snaps = len(get(f"https://www.tng-project.org/api/{simulation}/snapshots/"))
halo_url = (
    f"https://www.tng-project.org/api/TNG50-1/snapshots/{snapnum}/halos/{haloID}/"
)

# Try this in case of error: {'gas':'ElectronAbundance,InternalEnergy,Density,Masses'}
cutout_query = {
    "gas": "ElectronAbundance,InternalEnergy,Density,Masses,GFM_CoolingRate"
}
cutout = get(f"{halo_url}/cutout.hdf5", cutout_query)

redshift = np.array(
    get(f"https://www.tng-project.org/api/{simulation}/snapshots/")[snapnum]["redshift"]
)
a = 1 / (1 + redshift)
ckpc = UnitLength / a
UnitDensity = (UnitMass / h) / (ckpc / h) ** 3

with h5py.File(io.BytesIO(cutout.content), "r") as hdf:
    Density = np.array(hdf["PartType0/Density"])
    InternalEnergy = np.array(hdf["PartType0/InternalEnergy"])
    ElectronAbundance = np.array(hdf["PartType0/ElectronAbundance"])
    Lambda = np.array(hdf["PartType0/GFM_CoolingRate"])
    Masses = np.array(hdf["PartType0/Masses"])
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
    Density *= UnitDensity

with h5py.File(f"halo-prop_ID={haloID}.hdf5", "w") as store:
    store.create_dataset("Redshift", data=redshift)
    store.create_dataset("Density", data=Density)
    store.create_dataset("NumberDensity", data=NumberDensity)
    store.create_dataset("Temperature", data=Temperature)
    store.create_dataset("Volume", data=Volume)
    store.create_dataset("mu", data=mu)
    store.create_dataset("nH", data=NumberDensity * mu * XH * (mp / mH))
    store.create_dataset("Lambda", data=Lambda)
