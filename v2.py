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

#//
#parameter definitions
#(will go here eventually)
#egfr
Ae = 2.2
nae = 8.0
De = 0.16
Te = 2.0
#yan
Uy = 1.0
nuy = 2.0
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

lubensky = np.load("sims/half-success.npy")

nmol = 3
xdim = 13
ydim = 50

#//
# note here that nmol does not include the molecules
# in the lubensky model (ato, sense, u, and h)
def f(y, t, nmol, triang):
    #reshape flat array into shape (nmol,x,y)
    c = np.reshape( y, [ nmol, xdim , ydim ])
    #make empty array to hold rates of change
    xprime = np.empty(c.shape)

    e = c[0,:,:]
    y = c[1,:,:]
    p = c[2,:,:]

    #get values for a and u from loaded lubensky results
    #print t
    a = lubensky[t][0]
    u = lubensky[t][2]

    #actually calculate the rates of change

    #for 'egfr', the molecule at index [0,:,:]
    xprime[0,:,:] = hill(a/Ae,1.0,nae) - e + De * diff(e,triang) / Te
    # for 'y', the molecule at index [1,:,:]
    xprime[1,:,:] = hill(u/Uy,1.0,nuy) - y - hill(p/Py,1.0,npy) - hill(a/Ay,1.0,nay)
    # for 'p', the molecule at index [2,:,:]
    xprime[2,:,:] = hill(e/Ep,1.0,nep) - p - hill(y/Yp,1.0,nyp) - hill(p/Ap,1.0,nap)
    print type(xprime.flatten[0][0][0])
    return xprime.flatten

#//

distlattice = mk_rand_lattice(xdim,ydim) #set up the cells
triang = package(distlattice) #triangulate

#//

#set up initial conditions
initial = np.zeros((nmol,xdim,ydim))
timerange = range(0,140) #range to solve on
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

#//
