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
import numpy as np
from multiprocessing import Pool
from schwimmbad import MPIPool

def f(p, z, bm, bf):
    c=1 
    x = p[0]
    U = p[1]
    dx = U     
    dU =-2.*U/((1.+z))+ ((1.+(.73)/(.3*(1+z)**3+.73))*(3.-((1.+z)*U)**2*(1+z)**2*U))/(2*(1+z)**4)+(bm*c*np.exp(-c*x)/(1+bm*c*np.exp(-c*x)))*(.3*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))+(40*bf*c*np.exp(-c*x)/(1+bf*c*np.exp(-c*x)))*(.04*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))
    return array([dx, dU], float) 


def alpha_var(bm,bf,phi_1):
  phi_0 = 0
  z0 =0           
  zf = 3      
  N = 20000    
  c=1
  h = (zf - z0)/N
  delta_alpha= []
  zpoints = np.arange(z0, zf, h)
  xpoints = []
  vpoints = []
  p = array([phi_0, phi_1], float)
  for z in zpoints:
      xpoints.append(p[0])
      vpoints.append(p[1])
      k1 = h * f(p, z, bm, bf)
      k2 = h * f(p + 0.5*k1, z + 0.5*h, bm, bf)
      k3 = h * f(p + 0.5*k2, z + 0.5*h, bm, bf)
      k4 = h * f(p + k3, z + .5*h, bm, bf)
      p = p + (k1 + 2*k2 + 2*k3 + k4)/6
        
    
    
  for i in range(0,20000):
      alpha = bf*(np.exp(-c*xpoints[0])-np.exp(-c*xpoints[i]))/(1-bf*np.exp(c*xpoints[0]))
      delta_alpha.append(alpha)

  return alpha


QSO_data= [ -0.0000875,-0.0000701,-0.0000576,-0.0000450, -0.0000285]



def log_likelihood(theta, data):
    bm, bf, phi_1 = theta
    model = alpha_var(bm,bf, phi_1)
    arr= []

    if np.isnan(model):
      return  -1e6
    for i in range(0, len(data)):
        x= ((data[i]-model)/data[i])**2
        arr.append(x)
    return -.5*np.sum(arr)

bm_guess = 0
bf_guess = 0
phi_1_guess = 0
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([bm_guess,bf_guess, phi_1_guess]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=([QSO_data]))
bm_ml, bf_ml,phi_1_ml = soln.x

def log_prior(theta):
    bm, bf, phi_1 = theta
    if bm!=-1 and bf!=1 and -5 < bm < 5 and -5 < bf < 5 and -5 < phi_1< 5:
        return 0.0
    return -np.inf

def log_probability(theta, QSO_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, QSO_data)


with MultiPool() as pool:
	pos =  soln.x + 1e-3 * np.random.randn(32, 3)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,  args=([QSO_data]), pool=pool)
	sampler.run_mcmc(pos,30000, progress=True);



for i in range(0,32):
  chain= []
  chain= sampler.get_chain()[:,i,:]
  chain= np.concatenate((np.zeros((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  chain= np.concatenate((np.ones((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  np.savetxt('qtest_'+str(i)+'.txt',chain)


fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["bm", "bf","phi_1"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig("qplot.pdf")


tau = sampler.get_autocorr_time()
print(tau)


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

labels = ["bm", "bf","phi_1"]
fig = corner.corner(
    flat_samples, labels=labels,levels=(0.68,0.95,0.99,) );
fig.savefig("qpcont.pdf")
