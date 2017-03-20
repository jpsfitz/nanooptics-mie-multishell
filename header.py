## ======== Header ======== ##
# load python modules, 
# set fundamental constants,
# set basic directories,  
# load graphics options, 
# load refractive index library,

# -------- Import modules --------
print("Loading import modules ...")

# math 
print("  ... math, cmath (local)")
from math import *
from cmath import *

# directory access, system variables, re-imports
print("  ... os, sys, types, importlib")
import os, sys, types, importlib

# time monitoring
print("  ... time, datetime")
import time, datetime

# parallel processing
print("  ... ipyparallel as ipp")
import ipyparallel as ipp

# Numerical Python with multidimensional arrays
print("  ... numpy as np")
import numpy as np

# Scientific python for bessel functions
print("  ... scipy as sp")
import scipy as sp
print("      scipy.special (local, for Bessel functions)")
from scipy.special import *

# Scientific python 
print("      scipy.interpolate as interpolate")
import scipy.interpolate as interpolate
print("      scipy.signal as signal")
import scipy.signal as signal

# python 2D plotting library
print("  ... matplotlib")
import matplotlib
print("      matplotlib.pyplot as plt")
import matplotlib.pyplot as plt
print("      matplotlib.cm as cm")
import matplotlib.cm as cm
print("      matplotlib.pylab as pylab")
import matplotlib.pylab as pylab

# symbolic algebraic manipulation & pretty math output
print('  ... sympy')
import sympy
sympy.init_printing

# peak detection
# https://bitbucket.org/lucashnegri/peakutils
# print("  ... peakutils")
# import peakutils

# ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# -------- Fundamental constants --------
print("Defining fundamental constants ...")
import scipy.constants as SI
global AvogadroN, cumfs, cnmfs, heVfs, hbareVfs, eMass0eVnmfs, eps0enmV

AvogadroN = SI.N_A # 1/mol, Avogadro's number
print("  ... Avogadro's number: \'AvogadroN\'")

cumfs = SI.c*1e-15*1e6 # micron/fs, vacuum speed of light
cnmfs = SI.c*1e-15*1e9 # nm/fs, vacuum speed of light
print("  ... speed of light: \'cumfs\',\'cnmfs\'")

heVfs = SI.codata.value('Planck constant in eV s')*1e15 # eV*fs, Planck's constant
hbareVfs = heVfs/(2*pi) # eV*fs, h-bar, reduced Planck's constant
print("  ... Planck's constant: \'heVfs\',\'hbareVfs\'")

eMass0eVnmfs = SI.codata.value('electron mass energy equivalent in MeV')*1e6/(cnmfs*cnmfs) # eV/c^2 (c in nm/fs) = eV*fs^2/nm^2, electron mass
print("  ... electron rest mass: \'eMass0eVnmfs\'")

eps0enmV = 1/(2*SI.alpha*heVfs*cnmfs) # e^2/(eV*nm), permittivity of free space
print("  ... permittivity of free space: \'eps0enmV\'")


# -------- Local modules --------
print("Loading analysis codes ... ")

# Optical data
import refractive_index_library as ri
ri.printCheck()

# Mie theory codes
import mie # General
mie.printCheck()


# -------- Announce ready --------
print("Ready player one.")