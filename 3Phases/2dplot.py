import numpy as np
import matplotlib.pyplot as plt

sig_h = 0.494
sig_w = 1.001
sig_c = 0.3

del_h = 1.209
del_w = 0.298
del_c = 3.063

c_h = -1.113
c_w = -1.197
c_c = 11.780

rho_u = 1e-3
T_u = 10**5.762 * np.exp(-sig_h**/2.)
T_w = 10**5.261 * np.exp(-sig_w**/2.)
T_c = 10**4.101 * np.exp(-sig_c**/2.)

def toplot(T):
    rho = (np.exp(c_c) / T_u**(del_c -1) * T**(del_c-1) + np.exp(c_w) / T_u**(del_w -1) * T**(del_w-1) + np.exp(c_h) / T_u**(del_h -1) * T**(del_h-1))
           
    return rho
    
T = np.logspace(4.0,6.8,100)
x_plot = toplot(T)

plt.plot(x_plot,T)
plt.yscale('log')
plt.xscale('log')
plt.show()    
