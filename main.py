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

# to initialize the ni values.

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
				
	return [ni,rhor]

# calculate the rho and u, from ni.
def get_rho_u(ni,cxi,cyi,wi,lx,ly,mu,cs):
	for i in range(lx):
		for j in range(ly):
			nsum = 0.0
			# for density.
			for k in range(9):
				nsum = nsum + ni[i,j,k]
				uxsum = uxsum + ni[i,j,k]*cxi[k]
				uysum = uysum + ni[i,j,k]*cyi[k]
			rhor[i,j] = nsum
			# for velocity.
			ux[i,j] = uxsum/rhor[i,j]
			uy[i,j] = uysum/rhor[i,j]
			#
	return [rhor,ux,uy]

# calculate the equilibrium distribution.
def calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi):
	cs2 = cs**2.0
	cs4 = cs**4.0
	#
	for i in range(lx):
		for j in range(ly):
			u2 = ux[i,j]**2.0 + uy[i,j]**2.0
			for k in range(9):
				udotc = ux[i,j]*cxi[k] + uy[i,j]*cyi[k]
				udotc2 = udotc*udotc
				#
				ni_eq[i,j,k] = rhor[i,j]*wi[k]*(1.0 + udotc/cs2 + udotc2/(2.0*cs4) - u2/(2.0*cs2))
	#
	return ni_eq

# calculate the force vector. fx,fy.
def calc_force_vector(lmda,lx,ly):
	for i in range(lx):
		for j in range(ly):
			fx[i,j] = -(2.0*np.pi*lmda*A/lx)*(np.sin(2.0*np.pi*i/lx))
			fy[i,j] = 0.0
	#
	return [fx,fy]

# calculate the force term, fi.
def get_fi(lmda,rhor,fx,fy,gamma,cs):
	fx = np.zeros((lx,ly))
	fy = np.zeros((lx,ly))
	#
	cs2 = cs**2.0
	cs4 = cs**4.0
	for i in range(lx):
		for j in range(ly):
			uxloc[i,j] = ux[i,j] + fx[i,j]
			uyloc[i.j] = uy[i,j] + fy[i,j]
			for k in range(9):
				udotc = uxloc[i,j]*cxi[k] + uyloc[i,j]*cyi[k]
				t1x = cxi[k]/cs2
				t2x = 0.5*(1.0+gamma)
				t3x = (udotc/cs4)*cxi[k]
				t4x = ux[i,j]/cs2
				trm1x = (t1x + t2x*(t3x - t4x))*fx[i,j]
				#
				t1y = cyi[k]/cs2
				t2y = 0.5*(1.0+gamma)
				t3y = (udotc/cs4)*cyi[k]
				t4y = uy[i,j]/cs2
				trm1y = (t1y + t2y*(t3y - t4y))*fy[i,j]
				#
				fi[i,j,k] = wi[k]*(trm1x + trm1y)
	#
	return fi
					
					
#==========================================================#
#=======  main program here ===============================#
#==========================================================#

# initialize the system, i.e., ni values.
[ni, rhor] = initialize_ni(lx,ly,n,lmda,wi,mu,cs)
[rhor,ux,uy] = get_rho_u(ni,cxi,cyi,wi,lx,ly,mu,cs)
ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)



