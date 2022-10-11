import numpy as np
import random
import tqdm
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')
#import corner
from scipy.interpolate import interp1d
from numpy import array, arange
from scipy.optimize import minimize
#import emcee
from numpy import genfromtxt
import sys
# %matplotlib inline

z = np.loadtxt("EM_1.dat")[:, 0]
alpha = np.loadtxt("EM_1.dat")[:, 1]
z2 = np.loadtxt("EM_2.dat")[:, 0]
alpha2 = np.loadtxt("EM_2.dat")[:, 1]
z3 = np.loadtxt("EM_3.dat")[:, 0]
alpha3 = np.loadtxt("EM_3.dat")[:, 1]

interpolation= interp1d(z,alpha, fill_value='extrapolate' )
interpolation2= interp1d(z2,alpha2, fill_value='extrapolate' )
interpolation3= interp1d(z3,alpha3, fill_value='extrapolate' )

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

def dU_dx(U, x,m,w):
                    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
        return array( [U[1], -(2./(1.+x)+(3.*.3/(2.*(1.+x)**2.*(.3*(1.+x)**3.+.7))))*U[1] +2.*w*.3*np.exp(-2.*U[0])*((1.+x)/(.3*(1.+x)**3.+.7))-(np.power(10.,m))*U[0]/((x+1.)**2.*((1.+x)**3.+.7))],float)

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

def rk8(m,zeta):
  phi_0 = 0
  phi_1 = 0

  z0 = 0
  zf = 4000
  N = 75000
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

  return [xpoints, vpoints, zpoints]

def rho_cal( zeta,m):

  phi,phi_prime,z =rk8(m ,zeta )
  ys = phi


  interpolated_alpha= interp1d(z,ys, fill_value='extrapolate' )

  def rhoi1(redshift):
      return interpolated_alpha(redshift)*interpolation(redshift)
  def rhoi2(redshift):
      return interpolated_alpha(redshift)*interpolation2(redshift)
  def rhoi3(redshift):
      return interpolated_alpha(redshift)*interpolation3(redshift)

  rho_1= simpson(0,4000,rhoi1,5000)
  rho_2= simpson(0,4000,rhoi2,5000)
  rho_3= simpson(0,4000,rhoi3,5000)


  rho_val= np.transpose([rho_1,rho_2,rho_3])
  return rho_val

l = np.loadtxt("alpha_cl1.dat")[:, 0]
cl_tt = np.loadtxt("alpha_cl1.dat")[:, 1]
cl2_tt = np.loadtxt("alpha_cl2.dat")[:, 1]
cl3_tt = np.loadtxt("alpha_cl3.dat")[:, 1]
fid_tt= np.loadtxt("fid_cl.dat")[:, 1]

l = np.loadtxt("alpha_cl1.dat")[:, 0]
cl_ee = np.loadtxt("alpha_cl1.dat")[:, 2]
cl2_ee = np.loadtxt("alpha_cl2.dat")[:, 2]
cl3_ee = np.loadtxt("alpha_cl3.dat")[:, 2]
fid_ee= np.loadtxt("fid_cl.dat")[:, 2]

cl_te = np.loadtxt("alpha_cl1.dat")[:, 3]
cl2_te = np.loadtxt("alpha_cl2.dat")[:, 3]
cl3_te = np.loadtxt("alpha_cl3.dat")[:, 3]
fid_te= np.loadtxt("fid_cl.dat")[:, 3]

cl_tt_pc1= (1/np.sqrt(.088)**2)*(fid_tt-cl_tt)
cl_tt_pc2= (1/np.sqrt(.147)**2)*(fid_tt-cl2_tt)
cl_tt_pc3= (1/np.sqrt(.394)**2)*(fid_tt-cl3_tt)

cl_te_pc1= (1/np.sqrt(.088)**2)*(fid_te-cl_te)
cl_te_pc2= (1/np.sqrt(.147)**2)*(fid_te-cl2_te)
cl_te_pc3= (1/np.sqrt(.394)**2)*(fid_te-cl3_te)

cl_ee_pc1= (1/np.sqrt(.088)**2)*(fid_ee-cl_ee)
cl_ee_pc2= (1/np.sqrt(.147)**2)*(fid_ee-cl2_ee)
cl_ee_pc3= (1/np.sqrt(.394)**2)*(fid_ee-cl3_ee)

rho =rho_cal(3.5e-05, 1)
rho_1sig =rho_cal(5e-5,1)
rho_2sig =rho_cal(-5e-5, 1)

dcl_tt_1nsig=[]
dcl_te_1nsig=[]
dcl_ee_1nsig=[]
l_list= []

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_tt_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_tt_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_tt_pc3[j]
    dcl_tt_1nsig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_te_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_te_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_te_pc3[j]
    dcl_te_1nsig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_ee_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_ee_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_ee_pc3[j]
    dcl_ee_1nsig.append((delta_cl+delta_cl2+delta_cl3))

dcl_tt_2nsig=[]
dcl_te_2nsig=[]
dcl_ee_2nsig=[]
l_list= []

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_tt_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_tt_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_tt_pc3[j]
    dcl_tt_2nsig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_te_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_te_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_te_pc3[j]
    dcl_te_2nsig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_ee_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_ee_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_ee_pc3[j]
    dcl_ee_2nsig.append((delta_cl+delta_cl2+delta_cl3))

dcl_tt_best=[]
dcl_te_best=[]
dcl_ee_best=[]
l_list= []

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho[0]*cl_tt_pc1[j]
    delta_cl2 = rho[1]*cl_tt_pc2[j]
    delta_cl3 = rho[2]*cl_tt_pc3[j]
    dcl_tt_best.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho[0]*cl_te_pc1[j]
    delta_cl2 = rho[1]*cl_te_pc2[j]
    delta_cl3 = rho[2]*cl_te_pc3[j]
    dcl_te_best.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho[0]*cl_ee_pc1[j]
    delta_cl2 = rho[1]*cl_ee_pc2[j]
    delta_cl3 = rho[2]*cl_ee_pc3[j]
    dcl_ee_best.append((delta_cl+delta_cl2+delta_cl3))

dcl_tt_1sig=[]
dcl_te_1sig=[]
dcl_ee_1sig=[]
l_list= []

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_tt_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_tt_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_tt_pc3[j]
    dcl_tt_1sig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_te_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_te_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_te_pc3[j]
    dcl_te_1sig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_1sig[0]*cl_ee_pc1[j]
    delta_cl2 = rho_1sig[1]*cl_ee_pc2[j]
    delta_cl3 = rho_1sig[2]*cl_ee_pc3[j]
    dcl_ee_1sig.append((delta_cl+delta_cl2+delta_cl3))

dcl_tt_2sig=[]
dcl_te_2sig=[]
dcl_ee_2sig=[]
l_list= []

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_tt_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_tt_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_tt_pc3[j]
    dcl_tt_2sig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_te_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_te_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_te_pc3[j]
    dcl_te_2sig.append((delta_cl+delta_cl2+delta_cl3))

for j in range(0, len(cl_tt_pc1)):
    delta_cl = rho_2sig[0]*cl_ee_pc1[j]
    delta_cl2 = rho_2sig[1]*cl_ee_pc2[j]
    delta_cl3 = rho_2sig[2]*cl_ee_pc3[j]
    dcl_ee_2sig.append((delta_cl+delta_cl2+delta_cl3))

fig, ax = plt.subplots(2,3, figsize = (15,9))

ax[0,0].semilogx(l, np.abs(dcl_tt_1sig), color='k', label= r'1$\sigma$')
ax[0,0].semilogx(l, np.abs(dcl_tt_2sig), color='g', label= r'2$\sigma$')
ax[0,0].semilogx(l, np.abs(dcl_tt_best), color='b', label= 'Bestfit')
ax[0,0].set_title(r'$\Delta TT$')
ax[0,0].legend()

ax[0,1].semilogx(l, np.abs(dcl_te_1sig), color='k')
ax[0,1].semilogx(l, np.abs(dcl_te_2sig), color='g')
ax[0,1].semilogx(l, np.abs(dcl_te_best), color='b')
ax[0,1].set_title(r'$\Delta TE$')
ax[0,2].semilogx(l, np.abs(dcl_ee_1sig), color='k')
ax[0,2].semilogx(l, np.abs(dcl_ee_2sig), color='g')
ax[0,2].semilogx(l, np.abs(dcl_ee_best), color='b')
ax[0,2].set_title(r'$\Delta EE$')

ax[1,0].semilogx(l, (dcl_tt_1sig/(fid_tt+dcl_tt_1sig)), color='k')
ax[1,0].semilogx(l, (dcl_tt_2sig/(fid_tt+dcl_tt_2sig)), color='g')
ax[1,0].semilogx(l, (dcl_tt_best/(fid_tt+dcl_tt_best)), color='b')


ax[1,0].set_title('Fractional error induced in TT')
ax[1,1].semilogx(l, (dcl_te_1sig/(fid_te+dcl_te_1sig)), color='k')
ax[1,1].semilogx(l, (dcl_te_2sig/(fid_te+dcl_te_2sig)), color='g')
ax[1,1].semilogx(l, (dcl_te_best/(fid_te+dcl_te_best)), color='b')


ax[1,1].set_title('Fractional error induced in TE')
ax[1,2].semilogx(l, (dcl_ee_1sig/(fid_ee+dcl_ee_1sig)), color='k')
ax[1,2].semilogx(l, (dcl_ee_2sig/(fid_ee+dcl_ee_2sig)), color='g')
ax[1,2].semilogx(l, (dcl_ee_best/(fid_ee+dcl_ee_best)), color='b')

ax[1,2].set_title(r'Fractional error induced in EE')
ax[1,2].set_xlabel(r'Multipole moment, $\ell$')
ax[1,1].set_xlabel(r'Multipole moment, $\ell$')
ax[1,0].set_xlabel(r'Multipole moment, $\ell$')
ax[0,0].set_ylabel(r'$ \mu K^2$')
plt.savefig('delta_cl.pdf')
