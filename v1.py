'''
this file will eventually be a simple model of our own design that accounts for
the Yan and Pnt distributions using known interactions as well as a mystery Pnt
activator. It currently generates R8 cells in a time dependant manner
'''

#//

#imports
from support import *
import numpy as np
import scipy as sp

#//
#parameter definitions

a_prod = 0.2
a_deg = 0.1

nmol = 1
xdim = 20
ydim = 20

#//

#inputs are levels of molecules (y) time (t), number of molecular players (n),
#and a triang with neighbor_indices
def f(y, t, nmol, triang):
    #this reshapes the flat y array into three dimensions: molecule, x, and y. c for concentrations I guess.
    c = np.reshape( y, [ nmol, xdim , ydim ])
    xprime = np.empty(c.shape) #array in which to store the calculated rates of change 

    #levels of molecular players
    a = c[0,:,:]
    s = c[1,:,:]

    #array in which to store the rates of change
    xprime[0,:,:] = 0

    #time dependant atonal
    #construct a test array that will return true only for indicies where we want an R8 to appear
    r8s = np.empty([xdim,ydim])
    r8s[:] = False
    r8s[0::8,0::8] = True
    r8s[4::8,4::8] = True
    #an array that is true if its time for an R8 to flip
    time = np.empty([xdim,ydim])
    time[:] = False
    #using t/30 makes a new row of R8s ~ every 120 time units,
    #so that they are aprrox equal to seconds
    time[:,0:int(t/30)] = True
    #test array for whether atonal production should be on in a cell
    test = np.logical_and(r8s,time)
    #this line is a pretty janky hack, I think it can be made better
    xprime[0,:,:][test] = np.ones([xdim,ydim])[test] * a_prod - a_deg * a[test]

    
    #have to flatten the output so that it can be input to the next iteration of the function
    return xprime.flatten() 

#//

distlattice = mk_rand_lattice(xdim,ydim) #set up the cells
triang = package(distlattice) #triangulate

#//

#set up initial conditions
initial = np.zeros((nmol,xdim,ydim))
timerange = range(0,149) #range to solve on
#view initial conditions
plt.clf()
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = initial[0], vmin = 0, vmax = 2, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.savefig("fig")

#//

#solve it
args = (nmol, triang) #arguments to be passed to f
sol = odeint(f, initial.flatten(), timerange, args)
print "done"

resols = [np.reshape(i, [nmol,xdim,ydim]) for i in sol]
#reshaping the solutions so that they are easy to plot and view.
#resols is shaped like [time][molecule][x][y]
#resols[19][3] #this shows the values in all cells of molecule a (at index 0) at time 0

#//

#make a pretty plot of results, control the time point and molecule using 'c'
plt.clf()
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = resols[290][0], vmin = 0, vmax = 2, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.savefig("fig")

#//
