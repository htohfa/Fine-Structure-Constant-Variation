#Adapted from Dan's Code

import numpy as np
import tempfile, os
tempdir = '../dilatonupdated/'
rootname = os.path.join(tempdir, 'test')
from getdist import loadMCSamples,paramnames
from getdist import plots, gaussian_mixtures,MCSamples

#print(filename)
#paramnames.ParamList(fileName=filename)
epsilon=0.3
samples = loadMCSamples(rootname,settings={'ignore_rows':epsilon})


ndim=4

g = plots.get_subplot_plotter(width_inch = 4)
g.plot_2d([samples], 'a', 'bf',filled="True")


g = plots.get_subplot_plotter(width_inch = 4)
g.plot_2d([samples], 'a', 'phi_0',filled="True")

g = plots.get_subplot_plotter(width_inch = 10)

g.triangle_plot(samples, ['a','bf','c','phi_0'],filled=True)
