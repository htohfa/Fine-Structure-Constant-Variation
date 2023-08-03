import numpy as np
import random
import tqdm
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')
import corner
from scipy.interpolate import interp1d
from numpy import array, arange
from scipy.optimize import minimize
import emcee
from numpy import genfromtxt
import sys
from schwimmbad import MPIPool


c1 = 1.0 / 2.0    
c2 = (7.0 + np.sqrt(21) ) / 14.0
c3= (7.0 - np.sqrt(21))/14.0

a21 =  1.0 / 2.0;
a31 =  1.0 / 4.0;
a32 =  1.0 / 4.0
a41 =  1.0 / 7.0
a42 = -(7.0 + 3.0 * np.sqrt(21) ) / 98.0
a43 =  (21.0 + 5.0 * np.sqrt(21) ) / 49.0
a51 =  (11.0 + np.sqrt(21) ) / 84.0
a53 =  (18.0 + 4.0 * np.sqrt(21) ) / 63.0
a54 =  (21.0 - np.sqrt(21) ) / 252.0
a61 =  (5.0 + np.sqrt(21) ) / 48.0
a63 =  (9.0 + np.sqrt(21) ) / 36.0
a64 =  (-231.0 + 14.0 * np.sqrt(21) ) / 360.0
a65 =  (63.0 - 7.0 * np.sqrt(21) ) / 80.0
a71 =  (10.0 - np.sqrt(21) ) / 42.0
a73 =  (-432.0 + 92.0 * np.sqrt(21) ) / 315.0
a74 =  (633.0 - 145.0 * np.sqrt(21) ) / 90.0
a75 =  (-504.0 + 115.0 * np.sqrt(21) ) / 70.0
a76 =  (63.0 - 13.0 * np.sqrt(21) ) / 35.0
a81 =  1.0 / 14.0
a85 =  (14.0 - 3.0 * np.sqrt(21) ) / 126.0
a86 =  (13.0 - 3.0 * np.sqrt(21) ) / 63.0
a87 =  1.0 / 9.0
a91 =  1.0 / 32.0
a95 =  (91.0 - 21.0 * np.sqrt(21) ) / 576.0
a96 =  11.0 / 72.0
a97 = -(385.0 + 75.0 * np.sqrt(21) ) / 1152.0
a98 =  (63.0 + 13.0 * np.sqrt(21) ) / 128.0
a10_1 =  1.0 / 14.0
a10_5 =  1.0 / 9.0
a10_6 = -(733.0 + 147.0 * np.sqrt(21) ) / 2205.0
a10_7 =  (515.0 + 111.0 * np.sqrt(21) ) / 504.0
a10_8 = -(51.0 + 11.0 * np.sqrt(21) ) / 56.0
a10_9 =  (132.0 + 28.0 * np.sqrt(21) ) / 245.0
a11_5 = (-42.0 + 7.0 * np.sqrt(21) ) / 18.0
a11_6 = (-18.0 + 28.0 * np.sqrt(21) ) / 45.0
a11_7 = -(273.0 + 53.0 * np.sqrt(21) ) / 72.0
a11_8 =  (301.0 + 53.0 * np.sqrt(21) ) / 72.0
a11_9 =  (28.0 - 28.0 * np.sqrt(21) ) / 45.0
a11_10 = (49.0 - 7.0 * np.sqrt(21) ) / 18.0
b1  = 9.0 / 180.0
b8  = 49.0 / 180.0
b9  = 64.0 / 180.0


def dU_dx(U, x,m,w):
                    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
        return array( [U[1], -(2./(1.+x)+(3.*.3/(2.*(1.+x)**2.*(.3*(1.+x)**3.+.7))))*U[1] +2.*w*.3*np.exp(-2.*U[0])*((1.+x)/(.3*(1.+x)**3.+.7))-(np.power(10.,m))*U[0]/((x+1.)**2.*((1.+x)**3.+.7))],float)




def rk8(m,zeta):
  phi_0 = 0
  phi_1 = 0
  #phi_0 = 0.002728293224780778
  #phi_1 = 2.858399743936276e-07
  z0 = 0           
  zf = 4    
  N = 65000   
  h = (zf - z0)/N
  delta_alpha= []
  zpoints = np.arange(z0, zf, h)
  xpoints = []
  vpoints = []
  p = array([phi_0, phi_1], float)
  c1h = c1 * h 
  c2h = c2 * h
  c3h = c3 * h
  for z in zpoints:

      xpoints.append(2*p[0])
      vpoints.append(p[1])
      k_1 =  h* dU_dx(p, z , m, zeta)
      k_2 =  h* dU_dx(p+ a21*k_1, z+ c1h, m, zeta)
      k_3 =  h* dU_dx(p+ ( a31 * k_1 + a32 * k_2 ) , z+ c1h, m, zeta)
      k_4 =  h* dU_dx(p+ ( a41 * k_1 + a42 * k_2 + a43 * k_3 ) , z+ c2h, m, zeta)
      k_5 =  h* dU_dx(p+ ( a51 * k_1 + a53 * k_3 + a54 * k_4 ) , z+ c2h, m, zeta)
      k_6 =  h* dU_dx(p+ ( a61 * k_1 + a63 * k_3 + a64 * k_4 + a65 * k_5 )  , z+ c1h, m, zeta)
      k_7 =  h* dU_dx(p+ ( a71 * k_1 + a73 * k_3 + a74 * k_4 + a75 * k_5 + a76 * k_6 ), z+ c3h, m, zeta)
      k_8 =  h* dU_dx(p+ ( a81 * k_1 + a85 * k_5 + a86 * k_6 + a87 * k_7 ), z+ c3h , m, zeta)
      k_9 =  h* dU_dx(p+ ( a91 * k_1 + a95 * k_5 + a96 * k_6+ a97 * k_7 + a98 * k_8 ), z+ c1h, m, zeta)
      k_10 = h* dU_dx(p+ ( a10_1 * k_1 + a10_5 * k_5 + a10_6 * k_6 + a10_7 * k_7 + a10_8 * k_8 + a10_9 * k_9 ), z+ c2h, m, zeta)
      k_11 =  h* dU_dx(p + ( a11_5 * k_5 + a11_6 * k_6 + a11_7 * k_7+ a11_8 * k_8 + a11_9 * k_9 + a11_10 * k_10 ), z+ h, m, zeta)

      p = p +  (b1 * k_1 + b8 * k_8 + b9 * k_9 + b8 * k_10 + b1 * k_11)

  return [xpoints, zpoints]



def alpha_var( zeta,m, z):
  alpha=[]
  def dU_dx(U, x):
                      # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return  [U[1], -(2./(1.+x)+(3.*.03/(2.*(1.+x)**2.*(.3*(1.+x)**3.+.7))))*U[1] +2.*zeta*.3*np.exp(-2.*U[0])*((1.+x)/(.3*(1.+x)**3.+.7))-(np.power(10.,m))*U[0]/((x+1.)**2.*((1.+x)**3.+.7))]
   
  ys,xs = rk8(m,zeta)

  interp= interp1d(xs,ys, fill_value='extrapolate')
  for i in range(0, len(z)):
    alpha.append(interp(z[i]))
  return alpha


z_obs = [3.02, 2.59, 1.35, 2.14, 1.84, 1.94, 1.77, 1.43, 1.92, 1.69, 1.86, 2.15, 2.28, 2.43] 
z_obs= np.array(z_obs)
obs_alpha_var= [-27.9*1e-6, 5.7*1e-6, -4*1e-6, 6.7*1e-6, 3.5*1e-6, 5.1*1e-6, 8.4*1e-6, -21.3*1e-6, 8.5*1e-6, 1.3*1e-6, -9.9*1e-6, 5.2*1e-6, 7.5*1e-6, -12.2*1e-6]
obs_alpha_var = np.array(obs_alpha_var)
error= [34.2*1e-6, 3.4*1e-6, 2.3*1e-6, 3.5*1e-6, 2.5*1e-6,4.4*1e-6, 4.4*1e-6, 3.6*1e-6, 3.8*1e-6, 2.4*1e-6, 4.9*1e-6, 4.3*1e-6, 3.7*1e-6, 3.8*1e-6]
error= np.array(error)

def log_likelihood(theta, z, mu_data, muerr):
    zeta,m = theta
    model = alpha_var( zeta,m, z)
    sigma2 = muerr**2 
    return -0.5 * np.sum((mu_data - model) ** 2 / sigma2 )

soln = [1.1199217341402902e-06, -0.008197291589334835]

def log_prior(theta):
    zeta, m = theta
    if -5 < m < 3 and -3 < zeta < 3:
        return 0.0
    return -np.inf



def log_probability(theta, z, mu_data, muerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta,z, mu_data, muerr)



with MPIPool() as pool:
	pos =  soln + 1e-4 * np.random.randn(32, 2)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(z_obs, obs_alpha_var,error), pool=pool)
	sampler.run_mcmc(pos,5000, progress=True);

for i in range(0,32):
  chain= []
  chain= sampler.get_chain()[:,i,:]
  chain= np.concatenate((np.zeros((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  chain= np.concatenate((np.ones((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  np.savetxt('test_'+str(i)+'.txt',chain)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["zeta", "m"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig("plot.pdf")


tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)


import corner
labels = ["zeta", "m"]
fig = corner.corner(
    flat_samples, labels=labels,levels=(0.68,0.95,0.99,)
    );
fig.savefig("pcont_bsbm.pdf")
