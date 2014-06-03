__author__ = 'Abel Antonio Fernandez Higuera <afernandezh@dfacinf.uho.edu.cu>'

# Creado para probar InfomaxICA implementado en Python con datasets simples.

from numpy import array, dot
import matplotlib.pyplot as plotter
from scipy.io import loadmat

from ica import infomax_ica

matlab_file = loadmat('../data/sounds.mat')

x = matlab_file['sounds']
result = infomax_ica(x=x, k=5)
plotter.plot(result.T, color='blue')
plotter.show()

