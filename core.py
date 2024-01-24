
# --- LIBRARIES AND SEED

import numpy as np                                  # ofc numpy
from scipy.signal import convolve                   # will later be shifted to oaconvolve (must zero pad, does method='full' zero pad?)
from scipy.ndimage import gaussian_filter           # only for ease of development, will also be shifted to oaconvolve with 1d kernels
import matplotlib.pyplot as plt                     # for plotting (duh)
from matplotlib.colors import LogNorm               # my coloring choices are certainly better than my life choices
import time                                         # helps in quick and dirty timing of convolutions

np.random.seed(1020)                                # for reproducibility


# --- KERNELS

# complex 3x3 Scharr kernel for grad
# TODO > shift to better accuracy 5x5 later
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]])


# --- MESH

mesh_size = 256
base = np.zeros((mesh_size, mesh_size))


# --- PARTICLES

num_particles = 16      # number of stars
max_mass = 10           # max mass


# --- PREPARE MESH

coords = np.random.randint(mesh_size, size=(num_particles, 2))
masses = np.random.randint(mesh_size, size=(num_particles, 1))

for coord, mass in zip(coords, masses):
    base[coord[0], coord[1]] += mass            # spawn random mass stars in random locations and embed to base

# TODO > when moving on to fluids, we will change this whole thing


# --- VARIABLES
    
scale = 1000                                    # just for scaling the arrows longer cuz otherwise they are hard to see


# --- CONVOLUTIONS

start_time = time.time()

mesh1 = gaussian_filter(base, sigma=4)                           # primary convolution
mesh1 = convolve(mesh1, scharr, mode='same', method='auto')      # taking gradient once only here

# rest of the convolutions will build upon the mesh1

mesh2 = gaussian_filter(mesh1, sigma=16)                          # successively convoluting with larger and smoother gaussian kernels
mesh3 = gaussian_filter(mesh2, sigma=64)
mesh4 = gaussian_filter(mesh3, sigma=mesh_size)                  # final convolution, kernel sigma as large as whole mesh

final_mesh = mesh1 + mesh2 + mesh3 + mesh4                       # add all to get the final force mesh

# TODO > several things needs to be done here -
# 0. What is optimum kernel size here?
# 1. The gaussian kernels must be separated and custom convolution functions must be written, which is not very hard
# 2. The weights in the final sum must be figured out, which idk how to do analytically
# 3. The progressive subsampling downscaling must be implemented here
# 4. The final coarse kernel must be 1/sqrt(x^2 + k), which is inseparable, and must use fftconvolve for this
# 5. Use sockets for distributed computing! It's surprisingly simple once you have figured out the small nuances with packet length etc

print(time.time() - start_time)


# --- PLOTTING

# TODO > here we shall use sockets too, and also draw a bruteforce force arrow to see how good the directional angular accuracy is

plt.subplot(1, 2, 1)
plt.imshow(np.absolute(final_mesh), norm=LogNorm(), cmap='viridis')         # draw a nice force magnitude heatmap
plt.colorbar()

# and this force arrow of one star only for your reference, note how matplotlib takes x axis as y axis and vice versa in first two arguments
# I have absolutely no idea what I am messing up here, but it somehow works
plt.arrow(coords[0, 1], coords[0, 0], -np.real(final_mesh[coords[0, 0], coords[0, 1]])*scale, -np.imag(final_mesh[coords[0, 0], coords[0, 1]])*scale, color='r', width=1)

plt.subplot(1, 2, 2)
plt.imshow(np.angle(final_mesh), cmap='hsv')                              # draw the force direction
plt.colorbar()

# show plot ofc
plt.show()

