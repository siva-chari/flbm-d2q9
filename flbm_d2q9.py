#==========================================================#
#=======  imports here ====================================#
#==========================================================#

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

from numba import jit

#==========================================================#
#=======  parameters here =================================#
#==========================================================#

# switching time --> Tau.
Tau = 100
nSwitchingExpts = 1000

forward = 1
backward = -1

switching = forward # for forward switching experiment.

if switching > 0:
    lmdi = 0.0
    lmdf = 1.0
else:
    lmdi = 1.0
    lmdf = 0.0

dlmd = (lmdf-lmdi)/Tau

A = 0.01 # amplitude of the potential, and the value used in the paper = 0.01.
lx = 100
ly = 10

n = 1000
mu = 10.0
cs = 1.0/np.sqrt(3.0)
dt = 1.0
dx =  1.0
mp = mu
gamma = 0.8
nStepsThermalize = 10 # I JUST REDUCED IT'S COUNT TO CHECK THE RESULT

rhor = np.zeros((lx,ly))
rhor1 = np.zeros((lx,ly))
ux = np.zeros((lx,ly))
uy = np.zeros((lx,ly))
ux1 = np.zeros((lx,ly))
uy1 = np.zeros((lx,ly))
wi = np.zeros(9)
wi = [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.]
cxi = np.zeros(9)
cyi = np.zeros(9)
cxi = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
cyi = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0]
ni = np.zeros((lx,ly,9))
ni_eq = np.zeros((lx,ly,9))
n1 = np.zeros((lx,ly,9))
fi = np.zeros((lx,ly,9))
xi = np.zeros(9)
uxloc = np.zeros((lx,ly))
uyloc = np.zeros((lx,ly))
phi_lmda = np.zeros((lx,ly))
fx = np.zeros((lx,ly))
fy = np.zeros((lx,ly))

out_work_file = "workValues.dat"    # This data file consists of the work ensemble, for a given Tau.
fout_w = open(out_work_file,'w+')

#==========================================================#
#=======  functions here ==================================#
#==========================================================#

# to initialize the ni values.

@jit
def initialize_ni(lx,ly,n,lmda,wi,mu,cs):
    #
    # first we need to obtain the phi_sum.
    # for initialization purpose.
    #
    cs2 = cs**2.0
    bta = 1.0/(mu*cs2)  
    phi_sum = 0.0
    for i in range(lx):
        for j in range(ly):
            phi_lmda[i,j] = lmda*A*(np.cos((2.0*np.pi*i)/lx) + 1.0)
            phi_sum = phi_sum + np.exp(-bta*phi_lmda[i,j])
    #
    # kB T = mu * cs^2. ==> bta = 1.0/(kB*T)
    #
    #   cs2 = cs**2.0
    #   bta = 1.0/(mu*cs2)
    for i in range(lx):
        for j in range(ly):
            rhor[i,j] = ( np.exp(-bta*phi_lmda[i,j]) / phi_sum )*n*lx*ly
            #
            # now we can initialize ni. by using the weights, wi.
            for k in range(9):
                ni[i,j,k] = rhor[i,j]*wi[k]
    #
    return ni

# calculate the rho and u, from ni.
@jit
def get_rho_u(ni,cxi,cyi,lx,ly):
    for i in range(lx):
        for j in range(ly):
            nsum = 0.0
            uxsum = 0.0
            uysum = 0.0
            # for density.
            for k in range(9):
                nsum = nsum + ni[i,j,k]
                uxsum = uxsum + ni[i,j,k] * cxi[k]
                uysum = uysum + ni[i,j,k] * cyi[k]
            rhor[i,j] = nsum
            # for velocity.
            ux[i,j] = uxsum/rhor[i,j]
            uy[i,j] = uysum/rhor[i,j]
    #
    return [rhor,ux,uy]

# calculate the equilibrium distribution.
@jit
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
@jit
def calc_force_vector(lmda,lx,ly):
    global A
    for i in range(lx):
        for j in range(ly):
            fx[i,j] = -(2.0*np.pi*lmda*A/lx)*(np.sin(2.0*np.pi*i/lx))
            fy[i,j] = 0.0
    #
    return [fx,fy]

# calculate the force term, fi.
@jit
def get_fi(lmda,rhor,ux,uy,fx,fy,gamma,cs,lx,ly):
    # 
    cs2 = cs**2.0
    cs4 = cs**4.0
    for i in range(lx):
        for j in range(ly):
            uxloc[i,j] = ux[i,j] + fx[i,j]*0.5
            uyloc[i,j] = uy[i,j] + fy[i,j]*0.5
            for k in range(9):
                udotc = uxloc[i,j]*cxi[k] + uyloc[i,j]*cyi[k]
                t1x = cxi[k]/cs2
                t2x = 0.5*(1.0+gamma)
                t3x = (udotc/cs4)*cxi[k]
                t4x = uxloc[i,j]/cs2
                trm1x = (t1x + t2x*(t3x - t4x))*fx[i,j]
                #
                t1y = cyi[k]/cs2
                t2y = 0.5*(1.0+gamma)
                t3y = (udotc/cs4)*cyi[k]
                t4y = uyloc[i,j]/cs2
                trm1y = (t1y + t2y*(t3y - t4y))*fy[i,j]
                #
                fi[i,j,k] = wi[k]*(trm1x + trm1y)
    #
    return fi

# update rule, for ni. the flbm rule.
@jit
def update_ni_flbm(ni,ni_eq,rhor,fi,gamma,lx,ly,mu,wi):
    #n1 = np.zeros((lx,ly,9))
    #
    for i in range(lx):
        for j in range(ly):
            #calculate the neighboring/streaming sites, to i,j.
            # neighboring spin values, along x-dimension.
            iup = (i+1) % lx
            idown = (i-1) % lx
            # along y-dimension
            jup = (j+1) % ly
            jdown = (j-1) % ly
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
            xi[0] = np.random.normal(0.0, np.sqrt((xi0)))
            xi[1] = np.random.normal(0.0, np.sqrt((xi1)))
            xi[2] = np.random.normal(0.0, np.sqrt((xi2)))
            xi[3] = np.random.normal(0.0, np.sqrt((xi3)))
            xi[4] = np.random.normal(0.0, np.sqrt((xi4)))
            xi[5] = np.random.normal(0.0, np.sqrt((xi5)))
            xi[6] = -(xi[0]+xi[1]+2.0*xi[2]+xi[3]+2.0*xi[4]+2.0*xi[5])*0.5
            xi[7] = (xi[1]+xi[2]-xi[3]+xi[4]+2.0*xi[5])*0.5
            xi[8] = -(xi[0]+2.0*xi[1]+xi[2]+xi[4]+2.0*xi[5])*0.5
            #
            n1[i, j, 0]         = ni_eq[i,j,0] + gamma*(ni[i,j,0] - ni_eq[i,j,0]) + xi[0] + rhor[i,j]*fi[i,j,0]
            n1[iup, j, 1]       = ni_eq[i,j,1] + gamma*(ni[i,j,1] - ni_eq[i,j,1]) + xi[1] + rhor[i,j]*fi[i,j,1]
            n1[i, jup, 2]       = ni_eq[i,j,2] + gamma*(ni[i,j,2] - ni_eq[i,j,2]) + xi[2] + rhor[i,j]*fi[i,j,2]
            n1[idown, j, 3]     = ni_eq[i,j,3] + gamma*(ni[i,j,3] - ni_eq[i,j,3]) + xi[3] + rhor[i,j]*fi[i,j,3]
            n1[i, jdown, 4]     = ni_eq[i,j,4] + gamma*(ni[i,j,4] - ni_eq[i,j,4]) + xi[4] + rhor[i,j]*fi[i,j,4]
            n1[iup, jup, 5]     = ni_eq[i,j,5] + gamma*(ni[i,j,5] - ni_eq[i,j,5]) + xi[5] + rhor[i,j]*fi[i,j,5]
            n1[idown, jup, 6]   = ni_eq[i,j,6] + gamma*(ni[i,j,6] - ni_eq[i,j,6]) + xi[6] + rhor[i,j]*fi[i,j,6]
            n1[idown, jdown, 7] = ni_eq[i,j,7] + gamma*(ni[i,j,7] - ni_eq[i,j,7]) + xi[7] + rhor[i,j]*fi[i,j,7]
            n1[iup, jdown, 8]   = ni_eq[i,j,8] + gamma*(ni[i,j,8] - ni_eq[i,j,8]) + xi[8] + rhor[i,j]*fi[i,j,8]
            #
    return n1


# update rule for free evolution/ thermalization.
@jit
def update_ni_flbm_thermalize(ni,ni_eq,rhor,gamma,lx,ly,mu,wi):
    #
    for i in range(lx):
        for j in range(ly):
            #calculate the neighboring/streaming sites, to i,j.
            # neighboring spin values, along x-dimension.
            iup = (i+1) % lx
            idown = (i-1) % lx
            # along y-dimension
            jup = (j+1) % ly
            jdown = (j-1) % ly
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
            xi[0] = np.random.normal(0.0, np.sqrt((xi0)))
            xi[1] = np.random.normal(0.0, np.sqrt((xi1)))
            xi[2] = np.random.normal(0.0, np.sqrt((xi2)))
            xi[3] = np.random.normal(0.0, np.sqrt((xi3)))
            xi[4] = np.random.normal(0.0, np.sqrt((xi4)))
            xi[5] = np.random.normal(0.0, np.sqrt((xi5)))
            xi[6] = -(xi[0]+xi[1]+2.0*xi[2]+xi[3]+2.0*xi[4]+2.0*xi[5])*0.5
            xi[7] = (xi[1]+xi[2]-xi[3]+xi[4]+2.0*xi[5])*0.5
            xi[8] = -(xi[0]+2.0*xi[1]+xi[2]+xi[4]+2.0*xi[5])*0.5
            #
            n1[i, j, 0]         = ni_eq[i,j,0] + gamma*(ni[i,j,0] - ni_eq[i,j,0]) + xi[0] 
            n1[iup, j, 1]       = ni_eq[i,j,1] + gamma*(ni[i,j,1] - ni_eq[i,j,1]) + xi[1] 
            n1[i, jup, 2]       = ni_eq[i,j,2] + gamma*(ni[i,j,2] - ni_eq[i,j,2]) + xi[2] 
            n1[idown, j, 3]     = ni_eq[i,j,3] + gamma*(ni[i,j,3] - ni_eq[i,j,3]) + xi[3] 
            n1[i, jdown, 4]     = ni_eq[i,j,4] + gamma*(ni[i,j,4] - ni_eq[i,j,4]) + xi[4] 
            n1[iup, jup, 5]     = ni_eq[i,j,5] + gamma*(ni[i,j,5] - ni_eq[i,j,5]) + xi[5] 
            n1[idown, jup, 6]   = ni_eq[i,j,6] + gamma*(ni[i,j,6] - ni_eq[i,j,6]) + xi[6] 
            n1[idown, jdown, 7] = ni_eq[i,j,7] + gamma*(ni[i,j,7] - ni_eq[i,j,7]) + xi[7] 
            n1[iup, jdown, 8]   = ni_eq[i,j,8] + gamma*(ni[i,j,8] - ni_eq[i,j,8]) + xi[8] 
            #
    return n1

# calculate the small word done, dw.
@jit(nopython=True)
def get_dw(rhor,dlmd,lx,ly):
    global A
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
    ni = initialize_ni(lx,ly,n,lmdi,wi,mu,cs)
    [rhor,ux,uy] = get_rho_u(ni,cxi,cyi,lx,ly)
    ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)
    
    # initial free evolution/ thermalization.
    for i in range(nStepsThermalize):
        n1 = update_ni_flbm_thermalize(ni,ni_eq,rhor,gamma,lx,ly,mu,wi)
        ni = n1
        [rhor,ux,uy] = get_rho_u(ni,cxi,cyi,lx,ly)
        ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)
    
    # start a single switching experiment.
    wsum = 0.0
    for itr in range(Tau):
        #fx = np.zeros((lx,ly))
        #fy = np.zeros((lx,ly))
        #lmda = lmdi + (itr+1)*dlmd
        lmda = lmdi + (lmdf - lmdi)*itr/Tau
        
        # calculate force vector.
        [fx,fy] = calc_force_vector(lmda,lx,ly)
        
        # get the fi values, as per lamda.
        fi = get_fi(lmda,rhor,ux,uy,fx,fy,gamma,cs,lx,ly)
        
        # update ni, as per update rule, with switching.
        n1 = update_ni_flbm(ni,ni_eq,rhor,fi,gamma,lx,ly,mu,wi)
        
        # calculate rho, u, updated values.
        [rhor1,ux1,uy1] = get_rho_u(n1,cxi,cyi,lx,ly)
        
        # calculate the dw, due to switching.
        dw = get_dw(rhor,dlmd,lx,ly)
        wsum = wsum + dw
        
        # calculate the new equilibrium distribution function.
        ni_eq = calc_ni_eq(rhor1,ux1,uy1,cs,lx,ly,wi)
        
        # update the old ni,rhor,ux,uy with new ones.
        rhor = rhor1
        ux = ux1
        uy = uy1
        ni = n1
        
        # thermalize again, for a few steps.
        for ii in range(10):
            n1 = update_ni_flbm_thermalize(ni,ni_eq,rhor,gamma,lx,ly,mu,wi)
            ni = n1
            [rhor,ux,uy] = get_rho_u(ni,cxi,cyi,lx,ly)
            ni_eq = calc_ni_eq(rhor,ux,uy,cs,lx,ly,wi)
    #
    wTotal = wsum   
    fout_w.write("{:.8f}\n".format(wTotal)) 
    fout_w.flush()

#================ closing all data files.
fout_w.close()


wval = np.loadtxt('workValues.dat')

plt.hist(wval, histtype='step', bins=30, linewidth=1.5)


