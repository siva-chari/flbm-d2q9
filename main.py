#==========================================================#
#=======  imports here ====================================#
#==========================================================#

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


#==========================================================#
#=======  parameters here =================================#
#==========================================================#

# switching time --> Tau.
Tau = 10.0

forward = 1
backward = -1

switching = forward	# for forward switching experiment.
lmdi = 0.0
lmdf = 1.0
dlmd = (lmdf-lmdi)/Tau

lx = 16
ly = 8

n = 1000
mu = 10.0
cs = 1.0/sqrt(3.0)
dt = 1.0
dx =  1.0
mp = 1.0
gamma = 0.75

rhor = np.zeros((lx,ly))
ux = np.zeros((lx,ly))
uy = np.zeros((lx,ly))
wi = np.zeros(9)
cxi = np.zeros(9)
cyi = np.zeros(9)
ni = np.zeros((lx, ly, 9))
ni_eq = np.zeros((lx, ly, 9))
fi = np.zeros((lx, ly, 9))
xi = np.zeros(9)
uxloc = np.zeros((lx, ly))
uyloc = np.zeros((lx, ly))
phi_lmda = np.zeros((lx,ly))


out_work_file = "workValues.dat" 	# This data file consists of the work ensemble, for a given Tau.


#==========================================================#
#=======  functions here ==================================#
#==========================================================#

def initialize_ni(lx,ly,n,lmda,wi,mu,cs):
	#
	# first we need to obtain the phi_sum.
	# for initialization purpose.
	#
	phi_sum = 0.0
	for i in range(lx):
		for j in range(ly):
			phi_lmda[i,j] = lmda*A*(np.cos(2.0*np.pi*i/lx) + 1.0)
			phi_sum = phi_sum + phi_lmda[i,j]
	#
	# kB T = mu * cs^2. ==> bta = 1.0/(kB*T)
	#
	cs2 = cs**2.0
	bta = 1.0/(mu*cs2)
	for i in range(lx):
		for j in range(ly):
			rhor[i,j] = np.exp(-bta*phi_lmda[i,j])/(phi_sum) * n*lx*ly
			#
			# now we can initialize ni. by using the weights, wi.
			for k in range(9):
				ni[i,j,k] = rhor[i,j]*wi[k]
				
	
			



#==========================================================#
#=======  main program here ===============================#
#==========================================================#
