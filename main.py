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
nSwitchingExpts = 10

forward = 1
backward = -1

switching = forward	# for forward switching experiment.
lmdi = 0.0
lmdf = 1.0
dlmd = (lmdf-lmdi)/Tau
A = 0.01 # amplitude of the potential, and the value used in the paper = 0.01.
lx = 16
ly = 8

n = 1000
mu = 10.0
cs = 1.0/sqrt(3.0)
dt = 1.0
dx =  1.0
mp = 1.0
gamma = 0.75
nStepsThermalize = 10000

rhor = np.zeros((lx,ly))
rhor1 = np.zeros((lx,ly))
ux = np.zeros((lx,ly))
uy = np.zeros((lx,ly))
wi = np.zeros(9)
cxi = np.zeros(9)
cyi = np.zeros(9)
ni = np.zeros((lx, ly, 9))
ni_eq = np.zeros((lx, ly, 9))
n1 = np.zeros((lx,ly,9))
fi = np.zeros((lx, ly, 9))
xi = np.zeros(9)
uxloc = np.zeros((lx, ly))
uyloc = np.zeros((lx, ly))
phi_lmda = np.zeros((lx,ly))


out_work_file = "workValues.dat" 	# This data file consists of the work ensemble, for a given Tau.
fout_w = open(out_work_file,'w+')

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

# update rule, for ni. the flbm rule.
def update_ni_flbm(ni,ni_eq,rhor,fi,gamma,lx,ly,mu,wi):
	n1 = np.zeros((lx,ly,9))
	#
	for i in range(lx):
		for j in range(ly):
			#calculate the neighboring/streaming sites, to i,j.
			# neighboring spin values, along x-dimension.
			iup = (i+1)%lx
			idown = (i-1)%lx
			# along y-dimension
			jup = (j+1)%ly
			jdown = (j-1)%ly
			# 
			# now, get the xi, random numbers.
			# independent random numbers, xi, i=0...5.
            xi0 = rhor[i,j]*mu*wi[0]*np.sqrt(1.0-gamma**2.0)
            xi1 = rhor[i,j]*mu*wi[1]*np.sqrt(1.0-gamma**2.0)
            xi2 = rhor[i,j]*mu*wi[2]*np.sqrt(1.0-gamma**2.0)
            xi3 = rhor[i,j]*mu*wi[3]*np.sqrt(1.0-gamma**2.0)
            xi4 = rhor[i,j]*mu*wi[4]*np.sqrt(1.0-gamma**2.0)
            xi5 = rhor[i,j]*mu*wi[5]*np.sqrt(1.0-gamma**2.0)
            # since we need only 6 random numbers, the above is enough.
            # use the above values, to generate the required normal
            # random numbers xi. i=6...8.
            xi[0] = np.random.normal(0.0, np.sqrt(xi0))
            xi[1] = np.random.normal(0.0, np.sqrt(xi1))
            xi[2] = np.random.normal(0.0, np.sqrt(xi2))
            xi[3] = np.random.normal(0.0, np.sqrt(xi3))
            xi[4] = np.random.normal(0.0, np.sqrt(xi4))
            xi[5] = np.random.normal(0.0, np.sqrt(xi5))
            xi[6] = -(xi[0]+xi[1]+2.0*xi[2]+xi[3]+2.0*xi[4]+2.0*xi[5])*0.5
            xi[7] = (xi[1]+xi[2]-xi[3]+xi[4]+2.0*xi[5])*0.5
            xi[8] = -(xi[0]+2.0*xi[1]+xi[2]+xi[4]+2.0*xi[5])*0.5
            #
			n1[i, j, 0] 		= ni_eq[i,j,0] + gamma*(ni[i,j,0] - ni_eq[i,j,0]) + xi[0] + rhor[i,j]*fi[i,j,0]
			n1[iup, j, 1] 		= ni_eq[i,j,1] + gamma*(ni[i,j,1] - ni_eq[i,j,1]) + xi[1] + rhor[i,j]*fi[i,j,1]
			n1[i, jup, 2] 		= ni_eq[i,j,2] + gamma*(ni[i,j,2] - ni_eq[i,j,2]) + xi[2] + rhor[i,j]*fi[i,j,2]
			n1[idown, j, 3]		= ni_eq[i,j,3] + gamma*(ni[i,j,3] - ni_eq[i,j,3]) + xi[3] + rhor[i,j]*fi[i,j,3]
			n1[i, jdown, 4] 	= ni_eq[i,j,4] + gamma*(ni[i,j,4] - ni_eq[i,j,3]) + xi[4] + rhor[i,j]*fi[i,j,4]
			n1[iup, jup, 5] 	= ni_eq[i,j,5] + gamma*(ni[i,j,5] - ni_eq[i,j,5]) + xi[5] + rhor[i,j]*fi[i,j,5]
			n1[idown, jup, 6]	= ni_eq[i,j,6] + gamma*(ni[i,j,6] - ni_eq[i,j,6]) + xi[6] + rhor[i,j]*fi[i,j,6]
			n1[idown, jdown, 7]	= ni_eq[i,j,7] + gamma*(ni[i,j,7] - ni_eq[i,j,7]) + xi[7] + rhor[i,j]*fi[i,j,7]
			n1[iup, jdown, 8]	= ni_eq[i,j,8] + gamma*(ni[i,j,8] - ni_eq[i,j,8]) + xi[8] + rhor[i,j]*fi[i,j,8]
			#
	return n1


# update rule for free evolution/ thermalization.
def update_ni_flbm_thermalize(ni,ni_eq,rhor,gamma,lx,ly,mu,wi):
	n1 = np.zeros((lx,ly,9))
	#
	for i in range(lx):
		for j in range(ly):
			#calculate the neighboring/streaming sites, to i,j.
			# neighboring spin values, along x-dimension.
			iup = (i+1)%lx
			idown = (i-1)%lx
			# along y-dimension
			jup = (j+1)%ly
			jdown = (j-1)%ly
			# 
			# now, get the xi, random numbers.
			# independent random numbers, xi, i=0...5.
            xi0 = rhor[i,j]*mu*wi[0]*np.sqrt(1.0-gamma**2.0)
            xi1 = rhor[i,j]*mu*wi[1]*np.sqrt(1.0-gamma**2.0)
            xi2 = rhor[i,j]*mu*wi[2]*np.sqrt(1.0-gamma**2.0)
            xi3 = rhor[i,j]*mu*wi[3]*np.sqrt(1.0-gamma**2.0)
            xi4 = rhor[i,j]*mu*wi[4]*np.sqrt(1.0-gamma**2.0)
            xi5 = rhor[i,j]*mu*wi[5]*np.sqrt(1.0-gamma**2.0)
            # since we need only 6 random numbers, the above is enough.
            # use the above values, to generate the required normal
            # random numbers xi. i=6...8.
            xi[0] = np.random.normal(0.0, np.sqrt(xi0))
            xi[1] = np.random.normal(0.0, np.sqrt(xi1))
            xi[2] = np.random.normal(0.0, np.sqrt(xi2))
            xi[3] = np.random.normal(0.0, np.sqrt(xi3))
            xi[4] = np.random.normal(0.0, np.sqrt(xi4))
            xi[5] = np.random.normal(0.0, np.sqrt(xi5))
            xi[6] = -(xi[0]+xi[1]+2.0*xi[2]+xi[3]+2.0*xi[4]+2.0*xi[5])*0.5
            xi[7] = (xi[1]+xi[2]-xi[3]+xi[4]+2.0*xi[5])*0.5
            xi[8] = -(xi[0]+2.0*xi[1]+xi[2]+xi[4]+2.0*xi[5])*0.5
            #
			n1[i, j, 0] 		= ni_eq[i,j,0] + gamma*(ni[i,j,0] - ni_eq[i,j,0]) + xi[0] 
			n1[iup, j, 1] 		= ni_eq[i,j,1] + gamma*(ni[i,j,1] - ni_eq[i,j,1]) + xi[1] 
			n1[i, jup, 2] 		= ni_eq[i,j,2] + gamma*(ni[i,j,2] - ni_eq[i,j,2]) + xi[2] 
			n1[idown, j, 3]		= ni_eq[i,j,3] + gamma*(ni[i,j,3] - ni_eq[i,j,3]) + xi[3] 
			n1[i, jdown, 4] 	= ni_eq[i,j,4] + gamma*(ni[i,j,4] - ni_eq[i,j,3]) + xi[4] 
			n1[iup, jup, 5] 	= ni_eq[i,j,5] + gamma*(ni[i,j,5] - ni_eq[i,j,5]) + xi[5] 
			n1[idown, jup, 6]	= ni_eq[i,j,6] + gamma*(ni[i,j,6] - ni_eq[i,j,6]) + xi[6] 
			n1[idown, jdown, 7]	= ni_eq[i,j,7] + gamma*(ni[i,j,7] - ni_eq[i,j,7]) + xi[7] 
			n1[iup, jdown, 8]	= ni_eq[i,j,8] + gamma*(ni[i,j,8] - ni_eq[i,j,8]) + xi[8] 
			#
	return n1

# calculate the small word done, dw.
def get_dw(rhor,dlmd):
	rsum = 0.0
	for i in range(lx):
		for j in range(ly):
			rsum = rsum + rhor[i,j]*(np.cos(2.0*np.pi*i/lx) + 1.0)
	dw = rsum*dlmd*A
	return dw

#==========================================================#
#=======  main program here ===============================#
#==========================================================#

# start a number of experiments.
for iexpt in range(nSwitchingExpts):
	#
	# initialize the system, i.e., ni values.
	[ni, rhor] = initialize_ni(lx,ly,n,lmdi,wi,mu,cs)
	[rhor,ux,uy] = get_rho_u(ni,cxi,cyi,wi,lx,ly,mu,cs)
	ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)
	# initial free evolution/ thermalization.
	for itr in range(nStepsThermalize):
		n1 = update_ni_flbm_thermalize(ni,ni_eq,rhor,gamma,lx,ly,mu,wi)
		ni = n1
		[rhor,ux,uy] = get_rho_u(ni,cxi,cyi,wi,lx,ly,mu,cs)
		ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)
	# start a single switching experiment.
	wsum = 0.0
	for itr in rane(Tau):
		lmda = lmdi + (itr+1)*dlmd
		# get the fi values, as per lamda.
		fi = get_fi(lmda,rhor,fx,fy,gamma,cs)
		# update ni, as per update rule, with switching.
		n1 = update_ni_flbm(ni,ni_eq,rhor,fi,gamma,lx,ly,mu,wi)
		# calculate rho, u, updated values.
		[rhor1,ux,uy] = get_rho_u(n1,cxi,cyi,wi,lx,ly,mu,cs)
		# calculate the dw, due to switching.
		dw = get_dw(rhor,dlmd)
		wsum = wsum + dw
		# calculate the new equilibrium distribution function.
		ni_eq = calc_ni_eq(rhor1,ux,uy,cs,lx,ly,wi)
		# update the old ni,rhor,ux,uy with new ones.
		rhor = rhor1
		ni = n1
	#
	wTotal = wsum	
	fout_w.write("{:.8f}\n".format(wTotal))	
	fout_w.flush()

#================ closing all data files.
fout_w.close()
