# -*- coding: utf-8 -*-
"""Runaway dilaton mcmc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iY25WLZ8EDgLmiCuUIUZTTS9URkjENhO
"""


# Commented out IPython magic to ensure Python compatibility.
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
from schwimmbad import MultiPool


z = np.loadtxt("EM_1.dat")[:, 0]
alpha = np.loadtxt("EM_1.dat")[:, 1]
z2 = np.loadtxt("EM_2.dat")[:, 0]
alpha2 = np.loadtxt("EM_2.dat")[:, 1]
z3 = np.loadtxt("EM_3.dat")[:, 0]
alpha3 = np.loadtxt("EM_3.dat")[:, 1]

redshift_data=[]
alpha_eigenvalue =[]
for i in range(0,len(z)):
    if z[i]<1600:
        alpha_eigenvalue.append(alpha[i])
        redshift_data.append(z[i])
redshift_data2=[]
alpha_eigenvalue2 =[]
for i in range(0,len(z2)):
    if z2[i]<1600:
        alpha_eigenvalue2.append(alpha2[i])
        redshift_data2.append(z2[i])

redshift_data3=[]
alpha_eigenvalue3 =[]
for i in range(0,len(z)):
    if z[i]<1600:
        alpha_eigenvalue3.append(alpha3[i])
        redshift_data3.append(z3[i])

np.mean(alpha_eigenvalue)

interpolation= interp1d(redshift_data,alpha_eigenvalue, fill_value='extrapolate' )
interpolation2= interp1d(redshift_data2,alpha_eigenvalue2, fill_value='extrapolate' )
interpolation3= interp1d(redshift_data3,alpha_eigenvalue3, fill_value='extrapolate' )

#comaparable dataset
redshift1=[]
rho1 =[]
for i in range(0,len(z)):
    if 1000<z[i]<1600:
        alpha_eigenvalue.append(alpha[i])
        redshift_data.append(z[i])
redshift_data2=[]
alpha_eigenvalue2 =[]
for i in range(0,len(z2)):
    if z2[i]<1600:
        alpha_eigenvalue2.append(alpha2[i])
        redshift_data2.append(z2[i])

redshift_data3=[]
alpha_eigenvalue3 =[]
for i in range(0,len(z)):
    if z[i]<1600:
        alpha_eigenvalue3.append(alpha3[i])
        redshift_data3.append(z3[i])

def rho_cal( zeta,m):
  w= -.00023 #reference Value
  def dU_dx(U, x):
                    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return  [U[1], -(2./(1.+x)+(3.*.03/(2.*(1.+x)**2.*(.3*(1.+x)**3.+.7))))*U[1] +2.*w*.3*np.exp(-2.*U[0])*((1.+x)/(.3*(1.+x)**3.+.7))-(np.power(10.,m))*U[0]/((x+1.)**2.*((1.+x)**3.+.7))]
  U0 = [0., 0.]
  xs = np.linspace(0., 1600., 200)
  Us = odeint(dU_dx, U0, xs)
  ys = 2*Us[:,0]

  #ref_cases
  trapezoid_area=[]
  rho_ref=[]
  for k in range (0,199):
    kth = interpolation(xs[k])*ys[k]
    kplus1th = interpolation(xs[k+1])*ys[k+1]
    area= .5*(kth+ kplus1th)*(xs[k+1]-xs[k])
    trapezoid_area.append(area)
  for i in range (0,len(trapezoid_area)-1):
    integral_ref = trapezoid_area[i]+trapezoid_area[i+1]
  normalized_rho_ref = integral_ref/(-.00023)
  trapezoid_area=[]
  rho_ref2=[]
  for k in range (0,199):
    kth = interpolation2(xs[k])*ys[k]
    kplus1th = interpolation2(xs[k+1])*ys[k+1]
    area= .5*(kth+ kplus1th)*(xs[k+1]-xs[k])
    trapezoid_area.append(area)
  for i in range (0,len(trapezoid_area)-1):
    integral_ref2 = trapezoid_area[i]+trapezoid_area[i+1]
  normalized_rho_ref2 = integral_ref2/(-.00023)

  trapezoid_area=[]
  rho_ref3=[]
  for k in range (0,199):
    kth = interpolation3(xs[k])*ys[k]
    kplus1th = interpolation3(xs[k+1])*ys[k+1]
    area= .5*(kth+ kplus1th)*(xs[k+1]-xs[k])
    trapezoid_area.append(area)
  for i in range (0,len(trapezoid_area)-1):
      integral_ref3 = trapezoid_area[i]+trapezoid_area[i+1]
  normalized_rho_ref3 = integral_ref3/(-.00023)

  delta_A= zeta+.00023
  rho_1 = delta_A*normalized_rho_ref
  rho_2 = delta_A*normalized_rho_ref2
  rho_3 = delta_A*normalized_rho_ref3
  rho_val= np.transpose([rho_1,rho_2,rho_3])
  return rho_val




rho_dat= [.017,.039,.062 ]

sigma = [.088, .147, .394]



def log_likelihood(theta, rho_data, rhoerr):
    zeta,m = theta
    model = rho_cal(zeta,m)
    sigma2 = [[1/rhoerr[0] ** 2,0,0], [0,1/rhoerr[1]**2,0], [0,0, 1/rhoerr[2]**2]]
    rho_matrix= [(model[0]- .017),
                 (model[1]- .039),
                 (model[2]- .062)]
    chi= np.matmul(np.matmul(np.transpose(rho_matrix),sigma2),rho_matrix)
    #talk to dan about the normalization term
    return -.5*chi



zeta_guess = -.0004
m_guess= 0
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([zeta_guess,m_guess]) + .1 * np.random.randn(2)
soln = minimize(nll, initial, args=(rho_dat, sigma))
zeta_ml, m_ml = soln.x
print(zeta_ml,m_ml)


def log_prior(theta):
    zeta, m = theta
    if -5 < m < 5 and -.1 < zeta < .1:
        return 0.0
    return -np.inf


def log_probability(theta, rho_data, rho_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, rho_data, rho_err)

with MultiPool() as pool:
	pos =  soln.x + 1e-2 * np.random.randn(32, 2)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rho_dat, sigma), pool=pool)
	sampler.run_mcmc(pos,10000, progress=True);


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



for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    print(txt)
