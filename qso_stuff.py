# -*- coding: utf-8 -*-
"""QSO stuff.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18mU9g-r4KGjMbcM-TtP5MvV4gNiFxpeF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy import array, arange
from scipy.optimize import minimize

def f(p, z, a, c):
    x = p[0]
    U = p[1]
    dx = U     
    dU = -2.*U/((1.+z))+ ((1.+(.73)/(.3*(1+z)**3+.73))*(3.-((1.+z)*U)**2*(1+z)**2*U))/(2*(1+z)**4)+(np.exp(-c*(x-p[0]))*a*.3*(3.-((1.+z)*U)**2))/(2*(1+z)**4*(.04+.23+.73/((1+z)**3)+(2.7*10**-5*(1+7/8*(4/11)**(4/3)*2.99))*(1+z)**4))
    return array([dx, dU], float)

def alpha_var(a,bf,c,phi_1):
  phi_0 = 0
  z0 =0           
  zf = 2.6        
  N = 2000           
  h = (zf - z0)/N
  delta_alpha= []
  zpoints = np.arange(z0, zf, h)
  a_ref =10**-4
  xpoints = []
  vpoints = []
  p = array([phi_0, phi_1], float)
  for z in zpoints:
      xpoints.append(p[0])
      vpoints.append(p[1])
      k1 = h * f(p, z, a_ref, c)
      k2 = h * f(p + 0.5*k1, z + 0.5*h, a_ref, c)
      k3 = h * f(p + 0.5*k2, z + 0.5*h, a_ref, c)
      k4 = h * f(p + k3, z + .5*h, a_ref, c)
      p = p + (k1 + 2*k2 + 2*k3 + k4)/6
  for i in range(0,1999):
      alpha = bf*c**2*(c*xpoints[0]- c*xpoints[i])
      delta_alpha.append(alpha)
  return alpha

QSO_data= [ -0.0000875,-0.0000701,-0.0000576,-0.0000450, -0.0000285]

def log_likelihood(theta, data):
    a, bf, c, phi_1 = theta
    model = alpha_var(a,bf,c, phi_1)
    arr= []
    for i in range(0, len(data)):
      x= (data[i]-model)/model
      arr.append(x)
    return np.sum(arr)

a_guess = .2*10**-4
bf_guess = .01
c_guess = -1
phi_1_guess = 0
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_guess, bf_guess,c_guess, phi_1_guess]) + 0.1 * np.random.randn(4)
soln = minimize(nll, initial, args=(QSO_data))
a_ml, bf_ml, c_ml,phi_1_ml = soln.x
print(a_ml,bf_ml,c_ml, phi_1_ml)