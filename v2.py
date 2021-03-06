'''
This is my attempt to 'play back' the recorded output of the lubensky model onto
ours and use it to shape Yan and Pnt expression.
'''

#//

#imports
from support import *
import numpy as np
import scipy as sp

#//

def shRange(conc_array, triang):
    conc_list = conc_array.flatten(order ='A')
    val_array = np.zeros(conc_list.size)
    for pindex in range(len(triang.points)):
        for nindex in triang.neighbor_indices[pindex]:
            val_array[pindex] += conc_list[nindex]
    return np.reshape(val_array, np.shape(conc_array))
        
#//
#parameter definitions
#(will go here eventually)
#egfr/notch
Ae,An = .01,.01
nae,nan = 4.0,4.0

De,Dn = 1.0,1.0

Ee,Nn = 0.5,0.5
nee,nnn = 2.0,2.0

#yan
Ny = 0.5
nny = 2.0
Py = 1.0
npy = 2.0
Ay = 0.5
nay = 6.0

#pnt
Ep = 0.5
nep = 4.0
Yp = 0.5
nyp = 4.0
Ap = 0.5
nap = 6.0

lubensky = np.load("sims/second-half-success.npy")
lubensky = lubensky[:,:,:,15:]

nmol = 4
xdim = 13
ydim = 35

#//
# note here that nmol does not include the molecules
# in the lubensky model (ato, sense, u, and h)
def f(y, t, nmol, triang):
    #reshape flat array into shape (nmol,x,y)
    c = np.reshape( y, [ nmol, xdim , ydim ])
    #make empty array to hold rates of change
    xprime = np.empty(c.shape)

    e = c[0,:,:]
    n = c[1,:,:]
    y = c[2,:,:]
    p = c[3,:,:]

    #get values for a and u from loaded lubensky results
    a = lubensky[t][0]
    s = lubensky[t][1]

    #actually calculate the rates of change

    #for 'egfr', the molecule at index [0,:,:]
    xprime[0,:,:] = (1.0-hill(a/Ae,1.0,nae)) * (hill(e/Ee,1.0,nee) + De*shRange(s,triang)) - e + De*diff(e,triang)
    #for 'notch', the molecule at index [1,:,:]
    xprime[1,:,:] = (1.0-hill(a/An,1.0,nan)) * (hill(n/Nn,1.0,nnn) + Dn*shRange(s,triang)) - n + Dn*diff(n,triang)
    # for 'y', the molecule at index [1,:,:]
    xprime[2,:,:] = hill(n/Ny,1.0,nny)*(1-hill(a/Ay,1.0,nay))-y
    # for 'p', the molecule at index [2,:,:]
    xprime[3,:,:] = hill(e/Ep,1.0,nep)*(1-hill(a/Ap,1.0,nap))-p

    return xprime.flatten()

#//

distlattice = mk_rand_lattice(xdim,ydim) #set up the cells
triang = package(distlattice) #triangulate

#//

#set up initial conditions
initial = np.zeros((nmol,xdim,ydim))
timerange = range(0,10) #range to solve on
#view initial conditions
plt.clf()
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = initial[0], vmin = 0, vmax = 2, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.savefig("fig")

#//

#solve it
args = (nmol, triang)
sol = odeint(f,initial.flatten(),timerange,args)
print "done"

resols = [np.reshape(i, [nmol,xdim,ydim]) for i in sol]

#//

#make a pretty plot of results, control the time point and molecule using 'c'
plt.clf()
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = resols[1][0], vmin = 0, vmax = 0.2, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.savefig("fig")

#//
