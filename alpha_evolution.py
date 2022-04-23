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


def alpha_var(w,m):

    def dU_dx(U, x):
                    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
        return  [U[1], -(2./(1.+x)+(3.*.03/(2.*(1.+x)**2.*(.3*(1.+x)**3.+.7))))*U[1] +2.*w*.3*np.exp(-2.*U[0])*((1.+x)/(.3*(1.+x)**3.+.7))-(np.power(10.,m))*U[0]/((x+1.)**2.*((1.+x)**3.+.7))]
    U0 = [0., 0.]
    xs = np.linspace(0., 1600., 2000)
    Us = odeint(dU_dx, U0, xs)
    ys = 2*Us[:,0]
    return ys

def f(p, z, bm, bf):
    c=1
    x = p[0]
    U = p[1]
    dx = U
    dU =-2.*U/((1.+z))+ ((1.+(.73)/(.3*(1+z)**3+.73))*(3.-((1.+z)*U)**2*(1+z)**2*U))/(2*(1+z)**4)+(bm*c*np.exp(-c*x)/(1+bm*c*np.exp(-c*x)))*(.3*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))+(40*bf*c*np.exp(-c*x)/(1+bf*c*np.exp(-c*x)))*(.04*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(9.23*10**-5)*(1+z)**4))
    return array([dx, dU], float)


def alpha_var_dil(bm,bf,phi_1):
    phi_0 = 0
    z0 =0
    zf = 1600
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

    return delta_alpha


zpoints = np.linspace(0, 1600, 20000)
xs = np.linspace(0., 1600., 2000)
plt.plot(xs, alpha_var(.0001,0), label= 'BSBM')
plt.plot(zpoints,alpha_var_dil(0.323, 0.0017, 2.113497193975662), label= 'Runaway Dilaton')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\alpha$ variation, $\frac{\Delta \alpha}{\alpha}$')
plt.xlabel(r'Redshift, $ z$')
plt.savefig('alpha_evolution.pdf')
