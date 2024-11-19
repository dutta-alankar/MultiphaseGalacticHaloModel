# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:09:45 2023

@author: Alankar
"""
import numpy as np
import emcee
import corner
import h5py
import sys
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
import warnings

np.seterr(all='warn')

#def raise_on_warning(message, category, filename, lineno, file=None, line=None):
#    raise Exception(message)

#warnings.filterwarnings('always', raise_on_warning)
#warnings.simplefilter("error")    
def probT(y, params):
    fv    = params["fv"]
    sig   = params["sig"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    prob = np.array([(fv[i]/(2*np.pi*sig[i,0]*sig[i,1])) * \
                             np.sqrt(np.pi/A[i]) * \
                             np.exp( - ((C[i]-B[i]**2/(4*A[i]))*(y-ymed[i])**2 ))
            for i in range(fv.shape[0])])  
    return prob

def probRho(x, params):
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_rho"]
    A,B,C = params["factors"]
    
    prob = np.array([(fv[i]/(2*np.pi*sig[i,0]*sig[i,1])) * \
                             np.sqrt(np.pi/C[i]) * \
                             np.exp( - ((A[i]-B[i]**2/(4*C[i]))*(x-xmed[i])**2 ))
            for i in range(fv.shape[0])])
    return prob
# returning log10(p2D(x,y) * log(10)**2)
def log10prob2D(x, y, params):
    xx, yy = np.meshgrid(x, y)
    fV    = params["fV"]
    sig   = params["sig"]
    xmed  = params["median_rho"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    # print(fv, sig)
    lnprob = np.zeros((fV.shape[0],*xx.shape), dtype=np.float64)
    try:
        lnprob[:,:,:] = np.array([-np.log(fV[i]/(2*np.pi*sig[i,0]*sig[i,1]))  - ( 
                                                     A[i]*(xx-xmed[i])**2 +  
                                                     C[i]*(yy-ymed[i])**2 +
                                                     B[i]*(xx-xmed[i])*(yy-ymed[i]) )
            for i in range(fV.shape[0])])
        #lnprob[lnprob<=-300] = -300  
        #lnprob[lnprob>=300] = 300  
        lnprob = lnprob + 2.*np.log(np.log(10))
    except:
        print("Test")
        print(fV)
        print(sig)
        #sys.exit(1)
        return -np.inf
    log10prob = lnprob/np.log(10)
    return {"hot" : log10prob[0,:,:],
            "warm": log10prob[1,:,:],
            "cold": log10prob[2,:,:],
            }

bins = 100
halo_id = 110
try:
    file = h5py.File(f"Illustris-TNG50-1/halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

mass = np.log10(np.array(file["/NumberDensity"])) + np.log10(np.array(file["/Volume"]))
volume = np.log10(np.array(file["/Volume"]))
#sfr = np.array(file["/SFR"])
condition = np.logical_and(np.log10(file["/Temperature"]) >= 3.99, np.log10(file["/Temperature"]) <= 6.4)
condition = np.logical_and(condition, np.logical_and(np.log10(file["/nH"]) >= -6.0, np.log10(file["/nH"]) <= -1.5))

counts, xedges, yedges = np.histogram2d(
    x=np.log10(file["/nH"])[condition],
    y=np.log10(file["/Temperature"])[condition],
    weights=volume[condition],
    bins=(bins, bins),
    density=True,
)

x_data, y_data = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
min_val = np.min(counts[counts>0])
counts[counts<=min_val] = min_val
hist_data = np.log10(np.copy(counts))

Temperature = 10.**y_data # np.logspace( 3.9, 6.4, bins)
num_dens    = 10.**x_data # np.logspace(-6.8, 0.2, bins)
# print(Temperature)
# print(num_dens)

def gen_model(params):
    fV       = params["fV"]
    T_meds   = 10.**params["median_T"]
    rho_meds = 10.**params["median_rho"]
    sigs     = params["sig"]
    alphas     = params["alphas"]
        
    T_medV_u   = T_meds[0]
    rho_medV_u = rho_meds[0]
    
    cos_fac = np.cos(np.deg2rad(alphas))
    sin_fac = np.sin(np.deg2rad(alphas))

    A = np.array([ 0.5*(( cos_fac[i]/sigs[i,0])**2 + (sin_fac[i]/sigs[i,1])**2 )
                   for i in range(alphas.shape[0])])
    B = np.array([ sin_fac[i]*cos_fac[i]*(( 1./sigs[i,0])**2 - (1./sigs[i,1])**2 )
                   for i in range(alphas.shape[0])])
    C = np.array([ 0.5*(( cos_fac[i]/sigs[i,1])**2 + (sin_fac[i]/sigs[i,0])**2 )
                   for i in range(alphas.shape[0])])


    # alpha = np.arctan(1/(alphas-1))
    xi    = np.log(rho_meds/rho_medV_u)
    yi    = np.log(T_meds/T_medV_u)

    pdf_vol_rhoT = log10prob2D(np.log(num_dens/rho_medV_u), np.log(Temperature/T_medV_u),
                          {"fV": fV,
                           "sig": sigs,
                           "median_rho": xi,
                           "median_T": yi,
                           "factors": [A,B,C]
                          })

    hot_pdf  = pdf_vol_rhoT["hot"]
    warm_pdf = pdf_vol_rhoT["warm"]
    cold_pdf = pdf_vol_rhoT["cold"]
    '''
    hot_pdf[hot_pdf>0.2]  = 0.2
    hot_pdf[hot_pdf<-4.2] = -4.2
    
    warm_pdf[warm_pdf>0.2]  = 0.2
    warm_pdf[warm_pdf<-4.2] = -4.2
    
    cold_pdf[cold_pdf>0.2]  = 0.2
    cold_pdf[cold_pdf<-4.2] = -4.2
    '''
    total = np.log10(10.**hot_pdf + 10.**warm_pdf + 10.**cold_pdf)
    '''
    total[total>0.2]  = 0.2
    total[total<-4.2] = -4.2
    '''
    return total
    
def create_initial_walkers(params, nwalkers):
    fV       = params["fV"]
    T_meds   = params["median_T"]
    rho_meds = params["median_rho"]
    sigs     = params["sig"]
    alphas     = params["alphas"]
    
    walker_pos_ini = []
    # rng = np.random.default_rng(15248)  # can be called without a seed
    np.random.seed(15248)
    for i in range(nwalkers):
        fV_h = (fV[0] + 1e-3*np.random.randn(1))[0]
        fV_w = (fV[1] + 1e-3*np.random.randn(1))[0]
        fV_c = 1 - (fV_h + fV_w)
        fV_ini      = np.array([fV_h, fV_w, fV_c])
        T_med_ini   = T_meds   * (1+1e-2*np.random.randn(3))
        rho_med_ini = rho_meds * (1+1e-1*np.random.randn(3))
        sigs_ini    = sigs * (1+1e-2*np.random.randn(3,2))
        alphas_ini    = alphas * (1+1e-2*np.random.randn(3))
        
        walker_pos_ini.append([*fV_ini[:-1], 
                               *T_med_ini, 
                               *rho_med_ini,
                               *(sigs_ini.flatten()),
                               *alphas_ini,
                              ]
                             )
    return np.array(walker_pos_ini)

def ln_likelihood(params_array): # -ln(L2_norm)
    params = {"fV": np.hstack((params_array[:2], 1-np.sum(params_array[:2]))),
              "median_T": params_array[2:5],
              "median_rho": params_array[5:8],
              "sig": params_array[8:14].reshape((3,2)),
              "alphas": params_array[14:],
              }
    if ((params_array[0] + params_array[1])>1.0): # constraint on volume frac
        return -np.inf
    if (params_array[0]<0.7 or params_array[1]<0):
        return -np.inf
    if (params_array[8]<0  or params_array[8]>2.0 or params_array[9]<0 or params_array[9]>2.0 or #sigs hot
        params_array[10]<0 or params_array[10]>2.0 or params_array[11]<0 or  params_array[11]>2.0 or #sigs warm
        params_array[12]<0 or params_array[12]>2.0 or params_array[13]<0 or params_array[13]>2.0 or #sigs cold
        params_array[14]<5 or params_array[14]>80 or #angle hot
        params_array[15]<20 or params_array[15]>80 or #angle warm
        params_array[16]<120 or params_array[16]>180 or #angles cold
        params_array[5]>-3 or params_array[5]<-6 or   #den hot
        params_array[6]>-3 or params_array[6]<-6 or   #den warm
        params_array[7]>-1 or params_array[7]<-5 or  #den cold
        params_array[2]<5.3 or params_array[2]>6 or   # temp hot
        params_array[3]<4.4 or params_array[3]>5.6 or  # temp warm
        params_array[4]<3.99 or params_array[4]>4.5):  # temp cold
        return -np.inf
       
    model = gen_model(params)
    return -0.5 * (np.sum( ((hist_data-model).flatten())**2 ))

def ln_prior(params_array, params_init):
    fV       = params_init["fV"]
    T_meds   = params_init["median_T"]
    rho_meds = params_init["median_rho"]
    sigs     = params_init["sig"]
    alphas     = params_init["alphas"]
    prior = 0.
    
    if ((fV[0] + fV[1])>1.0): # constraint on volume frac
        return -np.inf
    if (fV[0]<0 or fV[1]<0 or fV[2]<0):
        return -np.inf
    if (sigs[0,0]<0 or sigs[0,1]<0 or
        sigs[1,0]<0 or sigs[1,1]<0 or
        sigs[2,0]<0 or sigs[2,1]<0 ): # postive sigs
        return -np.inf
    sp_prior = 0.5    
    spreads = np.array([sp_prior*fV[0],
                        sp_prior*fV[1],
                        sp_prior*T_meds[0],
                        sp_prior*T_meds[1],
                        sp_prior*T_meds[2],
                        sp_prior*rho_meds[0],
                        sp_prior*rho_meds[1],
                        sp_prior*rho_meds[2],
                        sp_prior*sigs[0,0], sp_prior*sigs[0,1],
                        sp_prior*sigs[1,0], sp_prior*sigs[1,1],
                        sp_prior*sigs[2,0], sp_prior*sigs[2,1],
                        sp_prior*alphas[0],
                        sp_prior*alphas[1],
                        sp_prior*alphas[2],
                       ])
    spreads = np.abs(spreads)
    means = np.hstack( (fV[:-1], T_meds, rho_meds, sigs.flatten(), alphas) )
    for i in range(len(spreads)):
        gauss_mu  = means[i]
        gauss_sig = spreads[i]
        x = params_array[i]
        prior += (-np.log(gauss_sig*np.sqrt(2*np.pi))-((x - gauss_mu)/(np.sqrt(2)*gauss_sig))**2)
    return prior

def ln_prob(params_array, params_init):
    lp = ln_prior(params_array, params_init)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(params_array)

data_dump = None
             
with open("params_data.pickle", "rb") as file_obj:
    data_dump = pickle.load(file_obj)

for key_val in data_dump.keys():
    exec(f"{key_val} = data_dump[\"{key_val}\"]")

A, B, C = ABC
xi, yi = xy_meds

T_meds = np.log10(T_meds)
rho_meds = np.log10(rho_meds)

'''    
f_Vh = 0.944
f_Vw = 0.055
f_Vc = 1. - (f_Vh + f_Vw)
fV = np.array([f_Vh, f_Vw, f_Vc])

T_h = np.log10(6.1e5)
T_w = np.log10(1.6e5)
T_c = np.log10(1.25e4)
T_meds = np.array([T_h, T_w, T_c])

rho_h = np.log10(1.1e-5)
rho_w = np.log10(3.3e-5)
rho_c = np.log10(1.5e-3)
rho_meds = np.array([rho_h, rho_w, rho_c])

# In rotated frame
sig_h = [0.75, 0.35]
sig_w = [0.9, 1.2]
sig_c = [1.1, 0.15]
sigs =  np.array([sig_h, sig_w, sig_c])

alphas = np.array([20,55,175])
'''
params_init = {"fV": fV,
               "median_T": T_meds,
               "median_rho": rho_meds,
               "sig": sigs, # 3X2
               "alphas": alphas,
              }

nwalkers = 50
ndim = 17
nsteps = 100000
walker_pos = create_initial_walkers(params_init, nwalkers)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, ln_prob, args=(params_init,)
)
sampler.run_mcmc(walker_pos, nsteps, progress=True)
'''
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                    args=(params_init,), 
                                    pool=pool)
    start = time.time()
    sampler.run_mcmc(walker_pos, nsteps, progress=True)
    end = time.time()
    multi_time = end - start
    print(f"Multiprocessing took {multi_time:.1f} seconds")
'''
fig, axes = plt.subplots(ndim, figsize=(35, 35), sharex=True)
samples = sampler.get_chain()
'''
with open("samples_arr.pickle", 'wb') as pfile:
    pickle.dump(samples, pfile, protocol=pickle.HIGHEST_PROTOCOL)
with open("sampler_obj.pickle", 'wb') as pfile:
    pickle.dump(sampler, pfile, protocol=pickle.HIGHEST_PROTOCOL)
'''
# np.save("mcmc_samples.npy", samples)
# np.save("mcmc_sampler.npy", sampler)
labels = [r"$f_V^{(h)}$", r"$f_V^{(w)}$", 
          r"$T_{med,V}^{(h)}$", r"$T_{med,V}^{(w)}$", r"$T_{med,V}^{(c)}$",
          r"$n_{med,V}^{(h)}$", r"$n_{med,V}^{(w)}$", r"$n_{med,V}^{(c)}$",
          r"$\sigma _1^{(h)}$", r"$\sigma _2^{(h)}$", 
          r"$\sigma _1^{(w)}$", r"$\sigma _2^{(w)}$",
          r"$\sigma _1^{(c)}$", r"$\sigma _2^{(c)}$",
          r"$\alpha ^{(h)}$", r"$\alpha ^{(w)}$", r"$\alpha ^{(c)}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig("walker_pos.png")

flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
print(flat_samples.shape)

fig = plt.figure(figsize=(35,35))
corner.corner(
    flat_samples, labels=labels,
    fig=fig,
)
plt.savefig("test_mcmc.png")

tau = sampler.get_autocorr_time()
print(tau)

