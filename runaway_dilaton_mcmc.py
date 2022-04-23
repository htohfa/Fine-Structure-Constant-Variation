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



z = np.loadtxt("EM_1.dat")[:, 0]
alpha = np.loadtxt("EM_1.dat")[:, 1]
z2 = np.loadtxt("EM_2.dat")[:, 0]
alpha2 = np.loadtxt("EM_2.dat")[:, 1]
z3 = np.loadtxt("EM_3.dat")[:, 0]
alpha3 = np.loadtxt("EM_3.dat")[:, 1]

interpolation= interp1d(z,alpha, fill_value='extrapolate' )
interpolation2= interp1d(z,alpha2, fill_value='extrapolate' )
interpolation3= interp1d(z,alpha3,fill_value='extrapolate' )

#the dynamic scalar field equation
def f(p, z, bm, bf):
    c=1
    x = p[0]
    U = p[1]
    dx = U
    dU =-2.*U/((1.+z))+ ((1.+(.73)/(.3*(1+z)**3+.73))*(3.-((1.+z)*U)**2*(1+z)**2*U))/(2*(1+z)**4)+(bm*c*np.exp(-c*x)/(1+bm*c*np.exp(-c*x)))*(.3*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))+(40*bf*c*np.exp(-c*x)/(1+bf*c*np.exp(-c*x)))*(.04*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))
    return array([dx, dU], float)

def simpson(a,b,f,N):
	h=(b-a)/(N)
	integral = f(a)+f(b)
	#initializing sum for even and odd
	even =0
	odd =0
	#initializing step sizes using 5.10 in book odd and even case
	n= a+h
	for i in range(1,int(N/2)+1):
		even = even+f(n)
		n= n+ 2*h
	n= a+2*h
	for i in range(1,int(N/2)):
		odd= odd+ float(f(n))
		n = n+2*h
	#returning the integral
	return (h/3)*(integral+2*even+4*odd)

def rho_cal(bm,bf,phi_1):
    phi_0 = 0
    z0 =0
    zf = 4000
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

   # print(np.shape(zpoints))
   # print(np.shape(delta_alpha))
    interpolated_alpha= interp1d(zpoints,delta_alpha, fill_value='extrapolate' )

    def rhoi1(redshift):
        return interpolated_alpha(redshift)*interpolation(redshift)
    def rhoi2(redshift):
        return interpolated_alpha(redshift)*interpolation2(redshift)
    def rhoi3(redshift):
        return interpolated_alpha(redshift)*interpolation3(redshift)

    rho_1= simpson(0,4000,rhoi1,5000)
    rho_2= simpson(0,4000,rhoi2,5000)
    rho_3= simpson(0,4000,rhoi3,5000)


    return [rho_1,rho_2,rho_3]


rho_dat= [.017,.039,.062 ]

sigma = [.088, .147, .394]



def log_likelihood(theta, rho_data, rhoerr):
    bm, bf, phi_1 = theta
    model = rho_cal(bm, bf, phi_1)
    for i in range(len(model)):
      if np.isnan(model[i]):
        return -1e6
    sigma2 = [[1/rhoerr[0] ** 2,0,0], [0,1/rhoerr[1]**2,0], [0,0, 1/rhoerr[2]**2]]
    rho_matrix= [(model[0]- .017),
                 (model[1]- .039),
                 (model[2]- .062)]
    chi= np.matmul(np.matmul(np.transpose(rho_matrix),sigma2),rho_matrix)
    #return chi*-.5 + log(sum (sigma2))
    #x= -.5* np.sum( np.matmul(rho_matrix,sigma2) + np.log(sigma2))
    return -.5*chi #talk to dan about the normalization term

bm_guess = 0
bf_guess = 0
phi_1_guess = 0
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([bm_guess,bf_guess, phi_1_guess]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(rho_dat, sigma))
bm_ml, bf_ml,phi_1_ml = soln.x
print(bm_ml,bf_ml, phi_1_ml)


def log_prior(theta):
    bm, bf, phi_1 = theta
    if bm!=-1 and bf!=1 and -5 < bm < 5 and -5 < bf < 5 and -5 < phi_1< 5:
        return 0.0
    return -np.inf


def log_probability(theta, rho_data, rho_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, rho_data, rho_err)

with MPIPool() as pool:
	pos =  soln.x + 1e-3 * np.random.randn(32, 3)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(rho_dat, sigma), pool=pool)
	sampler.run_mcmc(pos,45000, progress=True);

for i in range(0,32):
  chain= []
  chain= sampler.get_chain()[:,i,:]
  chain= np.concatenate((np.zeros((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  chain= np.concatenate((np.ones((len(sampler.get_chain()[:,0,0]),1), dtype=int), chain), axis=1)
  np.savetxt('test'+str(i)+'.txt',chain)




fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["bm", "bf", "phi_1"]
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
labels = ["a", "bf", "c", "phi_1"]
fig = corner.corner(
    flat_samples, labels=labels,levels=(0.68,0.95,0.99,) );
fig.savefig("pcont.pdf")



for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    print(txt)

chain = sampler.get_chain()[:, :, 0].T

plt.hist(chain.flatten(), 100)
plt.gca().set_yticks([])
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p(\theta)$");
plt.savefig("convergence.pdf")
