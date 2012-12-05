#//

#import 
import numpy as np
np.set_printoptions(suppress=True)
from scipy.integrate import odeint
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

#// 

#open both files in separate objects

res1 = np.load("/Users/james/Documents/research/code/dynamical/lubensky/sims/half-success.npy")
res2 = np.load("/Users/james/Documents/research/code/dynamical/lubensky/sims/second-half-success.npy")

#//

results = np.vstack((res1, res2))

#//

plt.scatter(distlattice[1].flatten(), distlattice[0].flatten(), c = results[45][0], vmin = 0, vmax = 1.7, s = 50)
plt.axes().set_aspect('equal')
plt.colorbar()
plt.show()

#//

%cd /