# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#imports
import numpy as np
np.set_printoptions(suppress=True)
from scipy.integrate import odeint
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# <codecell>

#functions

#hill function
def hill(x,k,n):
    return (x ** n) / (x ** n + k ** n)

#function for computing a lattice site from a pair of integers as in lubensky
def mksite(in_x, in_y):
        x = float(in_x) + .5 * float(in_y)
        y = (np.sqrt(3)/2) * float(in_y)
        return [x,y]
    
#function for computing a lattice site from a pair of integers, in a way that makes more sense for the rest of the program
def mksite2(in_x, in_y):
        x = float(in_x) + (.5 * (in_y%2))
        y = (np.sqrt(3)/2) * float(in_y)
        return [x,y]
    
#use that last function to compute the positions on a hexagonal lattice for x and y indicies
def mklattice(x,y):
    pos_array = np.empty([2,x,y])
    it = np.nditer(pos_array[0], flags=['multi_index'])
    while not it.finished:
        pos_array[0,it.multi_index[0],it.multi_index[1]] = mksite2(it.multi_index[0],it.multi_index[1])[0]
        pos_array[1,it.multi_index[0],it.multi_index[1]] = mksite2(it.multi_index[0],it.multi_index[1])[1]
        it.iternext()
    return pos_array

#same as above but with random jitter added to each position
def mk_rand_lattice(x,y):
    pos_array = np.empty([2,x,y])
    it = np.nditer(pos_array[0], flags=['multi_index'])
    while not it.finished:
        pos_array[0,it.multi_index[0],it.multi_index[1]] = mksite2(it.multi_index[0],it.multi_index[1])[0]
        pos_array[0,it.multi_index[0],it.multi_index[1]] +=np.random.normal(0,.15,1)
        pos_array[1,it.multi_index[0],it.multi_index[1]] = mksite2(it.multi_index[0],it.multi_index[1])[1]
        pos_array[1,it.multi_index[0],it.multi_index[1]] +=np.random.normal(0,.15,1)
        it.iternext()
    return pos_array


#find the neighbors of points in a Dealunay triangulation
def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.vertices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
            '''
            this is a one liner for 'if this simplex contains the point we're interested in,
            extend the neighbors list by appending all the *other* point indices in the simplex
            '''
    #Now we just have to strip out all the duplicate indices and return the list of negihbors:
    return list(set(neighbors))

#packages up a lattice as a triang with neighbors indexed in triang.neighbor_indicies, which is a dictionary for fast lookup
def package(lattice):
    triang = Delaunay(np.reshape(lattice.transpose(), [lattice[0,:,:].size, 2], order = "A"))
    triang.neighbor_indices = dict()
    for index,point in enumerate(triang.points):
        triang.neighbor_indices[index] = find_neighbors(index,triang)
    return triang

#the diffusion opeartor, as defined in lubensky
def diff(conc_array, triang):
    conc_list = conc_array.flatten(order ='A')
    val_array = np.zeros(conc_list.size)
    #an alternate implementation of the diffusion operator, this one using a precomputed triang
    for pindex in range(len(triang.points)):
        for nindex in triang.neighbor_indices[pindex]:
            val_array[pindex] += (1 / np.linalg.norm(triang.points[pindex] - triang.points[nindex])**2) * (conc_list[nindex] - conc_list[pindex])
    return np.reshape(val_array, np.shape(conc_array))

# <codecell>

#parameters. defined globally for now, there's probably a more 'best practices' way to do this but this is more syntactically terse so I'm leaving it like this for now
#values taken from lubensky
Aa=0.65
As=0.5
Ah=1.5
Au=2.2
S=0.57
Ts=4.0
na=4.0
ns=4.0
nh=8.0
nu=8.0
ms=4.0
U=0.000004
H=0.0088
#G=0.8
G=0.8
#Dh=2.0
Dh=200
Th=101.0
mu=6.0
mh=4.0
F=0.6
Du=0.16
Tu=2.0

#use xdim and y dim to control the size of the rectangular grid that is solved for
nmol = 4
xdim = 13
ydim = 50

# <codecell>

#inputs are levels of molecules (y) time (t), number of molecular players (n), and a triang with neighbor_indices
def f(y, t, nmol, triang):
    c = np.reshape( y, [ nmol, xdim , ydim ]) #this reshapes the flat y array into three dimensions: molecule, x, and y. c for concentrations I guess.
    xprime = np.empty(c.shape) #array in which to store the calculated rates of change
    
    #the indices that correspond to each molecule are: 0=a, 1=s, 2=u, 3=h, so lets make variables that we can use to refer to them convieniently:
    a = c[0,:,:]
    s = c[1,:,:]
    u = c[2,:,:]
    h = c[3,:,:]
    
    ###equations that actually calculate the rates of change###
    ###taken from lubensky et al PNAS 2010###
    
    #for 'a', the first molecule, at index [0,:,:]
    xprime[0,:,:] = hill(a/Aa, 1.0, na) - a  +  F*hill(s/S, 1.0, ms)  +  G*hill(h/H, 1.0, mh) * (1 - hill(u/U, 1.0, mu))
    
    #for 's', the molecule at index [1,:,:]
    xprime[1,:,:] = ( hill(a/As, 1.0, ns) - s ) / Ts
    
    #for 'u', the molecule at index [2,:,:]
    xprime[2,:,:] = ( hill(a/Au, 1.0, nu) - u + Du * diff(u, triang) ) / Tu
    
    #for 'h', the molecule at index [3,:,:]
    xprime[3,:,:] = ( hill(a/Ah, 1.0, nh) - h + Dh * diff(h, triang) ) / Th
    
    return xprime.flatten() #have to flatten the output so that it can be input to the next iteration of the function

# <codecell>

distlattice = mk_rand_lattice(xdim,ydim) #set up the cells
triang = package(distlattice) #triangulate

# <codecell>

args = (nmol, triang) #arguments to be passed to f
initial = np.zeros((nmol,xdim,ydim))
initial[0,0:30:8,0:30:8] = 1.5 #set up initial conditions
initial[0,4:30:8,4:30:8] = 1.5 #set up initial conditions
initial[3,0:30,0:30] = 0.005
timerange = range(0,150) #time to solve on
#make a pretty plot of inital conditions
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = initial[0], vmin = 0, vmax = 2, s = 20)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.draw()
plt.show()
plt.ion()

#
plt.clf()
# <Codecell|>
        
#solve it
sol = odeint(f, initial.flatten(), timerange, args)
print "done"
# <codecell>

resols = [np.reshape(i, [nmol,xdim,ydim]) for i in sol]
#reshaping the solutions so that they are easy to plot and view.
#resols is shaped like [time][molecule][x][y]
#resols[19][3] #this shows the values in all cells of molecule a (at index 0) at time 0

# <codecell>
#make a pretty plot of results, control the time point and molecule using 'c'
plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = resols[45][0], vmin = 0, vmax = 1.7, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.show()
initial = resols[149]
#
plt.clf()
#

np.save("second-half-succes",resols) #can save the results to a file for later analysis


#
resols = np.load("/Users/james/Documents/research/code/dynamical/lubensky/success.npy")
