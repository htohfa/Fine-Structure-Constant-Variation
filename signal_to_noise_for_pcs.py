import numpy as np
import random
import tqdm
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')

from scipy.interpolate import interp1d
from numpy import array, arange
from scipy.optimize import minimize

from numpy import genfromtxt
import sys


z = np.loadtxt("mode1.dat")[:, 0]
alpha = np.loadtxt("mode1.dat")[:, 1]
z2 = np.loadtxt("mode2.dat")[:, 0]
alpha2 = np.loadtxt("mode2.dat")[:, 1]
z3 = np.loadtxt("mode3.dat")[:, 0]
alpha3 = np.loadtxt("mode3.dat")[:, 1]
z4 = np.loadtxt("mode4.dat")[:, 0]
alpha4 = np.loadtxt("mode4.dat")[:, 1]
z5 = np.loadtxt("mode5.dat")[:, 0]
alpha5 = np.loadtxt("mode5.dat")[:, 1]
z6 = np.loadtxt("mode6.dat")[:, 0]
alpha6 = np.loadtxt("mode6.dat")[:, 1]
z7 = np.loadtxt("mode7.dat")[:, 0]
alpha7 = np.loadtxt("mode7.dat")[:, 1]
z8 = np.loadtxt("mode8.dat")[:, 0]
alpha8 = np.loadtxt("mode8.dat")[:, 1]
z9 = np.loadtxt("mode9.dat")[:, 0]
alpha9 = np.loadtxt("mode9.dat")[:, 1]
z10 = np.loadtxt("mode10.dat")[:, 0]
alpha10 = np.loadtxt("mode10.dat")[:, 1]


interpolation= interp1d(z,alpha, fill_value='extrapolate' )
interpolation2= interp1d(z2,alpha2, fill_value='extrapolate' )
interpolation3= interp1d(z3,alpha3, fill_value='extrapolate' )
interpolation4= interp1d(z4,alpha4, fill_value='extrapolate' )
interpolation5= interp1d(z5,alpha5, fill_value='extrapolate' )
interpolation6= interp1d(z6,alpha6, fill_value='extrapolate' )
interpolation7= interp1d(z7,alpha7, fill_value='extrapolate' )
interpolation8= interp1d(z8,alpha8, fill_value='extrapolate' )
interpolation9= interp1d(z9,alpha9, fill_value='extrapolate' )
interpolation10= interp1d(z10,alpha10, fill_value='extrapolate' )


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
  #phi_0 = 0.002728293224780778
  #phi_1 = 2.858399743936276e-07
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

phi,phi_prime,z =rk8(3,-6.8e-5)

def rho_cal( zeta,m):

	ys = phi

	interpolated_alpha= interp1d(z,ys, fill_value='extrapolate' )

	def rhoi1(redshift):
		return interpolated_alpha(redshift)*interpolation(redshift)
	def rhoi2(redshift):
		return interpolated_alpha(redshift)*interpolation2(redshift)
	def rhoi3(redshift):
		return interpolated_alpha(redshift)*interpolation3(redshift)
	def rhoi4(redshift):
		return interpolated_alpha(redshift)*interpolation4(redshift)
	def rhoi5(redshift):
        	return interpolated_alpha(redshift)*interpolation5(redshift)
	def rhoi6(redshift):
        	return interpolated_alpha(redshift)*interpolation6(redshift)
	def rhoi7(redshift):
        	return interpolated_alpha(redshift)*interpolation7(redshift)
	def rhoi8(redshift):
        	return interpolated_alpha(redshift)*interpolation8(redshift)
	def rhoi9(redshift):
		return interpolated_alpha(redshift)*interpolation9(redshift)
	def rhoi10(redshift):
		return interpolated_alpha(redshift)*interpolation10(redshift)

	rho_1= simpson(0,4000,rhoi1,5000)
	rho_2= simpson(0,4000,rhoi2,5000)
	rho_3= simpson(0,4000,rhoi3,5000)
	rho_4= simpson(0,4000,rhoi4,5000)
	rho_5= simpson(0,4000,rhoi5,5000)
	rho_6= simpson(0,4000,rhoi6,5000)
	rho_7= simpson(0,4000,rhoi7,5000)
	rho_8= simpson(0,4000,rhoi8,5000)
	rho_9= simpson(0,4000,rhoi9,5000)
	rho_10= simpson(0,4000,rhoi10,5000)





	rho_val= np.transpose([rho_1,rho_2,rho_3, rho_4,rho_5,rho_6,rho_7,rho_8,rho_9,rho_10])
	return rho_val



#sigma = [.006, .012, .036]
rho= rho_cal(-6.8e-5,3)
#phi,phi_prime,z =rk8(3,-6.8e-5)

sigma = [0.00134822, 0.00313751 ,0.00652159, 0.0129908, 0.0588853, 0.162472 ,0.2465, 0.434743, 0.524369, 0.916163]
#calculating risk factor



s_n=[]
for i in range(0, 10):
	sn = ((rho[i]/sigma[i])**2)**(1/2)
	s_n.append(sn)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.figsize"] = [8.0,6.0]
axislabelfontsize= 54

matplotlib.mathtext.rcParams['legend.fontsize']=26


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


plt.rc("axes", linewidth=2.0)
plt.rc("lines", markeredgewidth=3)
plt.rc('axes', labelsize=32)
plt.rc('xtick', labelsize = 32)
plt.rc('ytick', labelsize = 32)

fig_width_pt = 703.27  #513.17           # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean=0.9
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
#
params = {'backend': 'pdf',
             'axes.labelsize': 84,
             'lines.markersize': 4,
             'font.size': 100,
             'xtick.major.size':6,
             'xtick.minor.size':3,
             'ytick.major.size':6,
             'ytick.minor.size':3,
             'xtick.major.width':0.5,
             'ytick.major.width':0.5,
             'xtick.minor.width':0.5,
             'ytick.minor.width':0.5,
             'lines.markeredgewidth':1,
             }


matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.figsize"] = [8.0,6.0]

matplotlib.mathtext.rcParams['legend.fontsize']=14


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

n= [1,2,3,4,5,6,7,8,9,10]

plt.scatter(n,s_n, label= 'SO Forecast',marker='^', s=200)

#plt.xscale('log')
plt.xlabel('PC index')
plt.ylabel('S/N')
plt.tight_layout()
#plt.savefig('sn_so.pdf')



z = np.loadtxt("EM_1.dat")[:, 0]
alpha = np.loadtxt("EM_1.dat")[:, 1]
z2 = np.loadtxt("EM_2.dat")[:, 0]
alpha2 = np.loadtxt("EM_2.dat")[:, 1]
z3 = np.loadtxt("EM_3.dat")[:, 0]
alpha3 = np.loadtxt("EM_3.dat")[:, 1]



interpolation= interp1d(z,alpha, fill_value='extrapolate' )
interpolation2= interp1d(z2,alpha2, fill_value='extrapolate' )
interpolation3= interp1d(z3,alpha3, fill_value='extrapolate' )

phi,phi_prime,z =rk8(3,-6.8e-5)

def rho_cal( zeta,m):

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

rho= rho_cal(-6.8e-5,3)
sigma = [.006, .012, .036]


s_n=[]
for i in range(0, 3):
	sn = ((rho[i]/sigma[i])**2)**(1/2)
	s_n.append(sn)

n= [1,2,3]

interpolate = interp1d(n, s_n, fill_value='extrapolate' )


n_= [ 4,5,6,7,8,9,10]
s_n_=[]
for i in range(0,len(n_)):
	s_n_.append(s_n[2])

plt.scatter(n,s_n, label= 'Planck 2018',marker='*', s=200)
plt.scatter(n_,s_n_ ,marker='*',color='red', s=200)


plt.legend()
#plt.xscale('log')
plt.xlabel('PC index')
plt.ylabel('S/N')
plt.tight_layout()
plt.savefig('sn_cmb.pdf')
