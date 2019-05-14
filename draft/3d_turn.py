from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

mu = 0
a, b = -.5, .5
delta, m = 1, 2
alpha, off = 1, 1

def tilde(x, delta=0, m=0):
    return np.sqrt(2) * x - (m+delta)/np.sqrt(2)


def phi(t):
    return (1/np.sqrt(2 * np.pi)) * np.exp(-(t**2) / 2)

def PHI(x, loc=0, scale=1):
    return stats.norm.cdf(x, loc=loc, scale=scale)


def minmax(a,b,delta,m,alpha, off, mu):
    E1 = PHI(b- mu) - PHI(b-delta) + PHI(a-delta) - PHI(a-mu)
    E2 = (np.exp(-((m-delta)/2)**2)/np.sqrt(2)) * (off + (alpha -1 )*(PHI(tilde(b,delta=delta, m=m)) - 
                                                                      PHI(tilde(a,delta=delta, m=m))))
    return E1 - E2



X = np.arange(-1, 1, 0.01)
Y =  np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(X, Y)


Z = minmax(a,X,Y,m,alpha, off, mu)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
