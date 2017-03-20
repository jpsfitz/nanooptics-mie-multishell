# ======== Mie theory analysis codes ======
# Mie theory optical cross sections from a multi-layered spherical particle
# Reference: Moroz (2005)
#   http://dx.doi.org/10.1016/j.aop.2004.07.002  
#   http://www.wave-scattering.com/moroz.html


# ======== Header ========
# -------- Import modules --------
# math 
from math import *
from cmath import *

# directory access
import os

# Numerical Python with multidimensional arrays
import numpy as np

# Scientific python for bessel functions
import scipy as sp
from scipy.special import *

# Scientific python for interpolation
import scipy.interpolate as interpolate

# -------- Fundamental constants --------
import scipy.constants as SI
AvogadroN = SI.N_A # 1/mol, Avogadro's number
cumfs = SI.c*1e-15*1e6 # micron/fs, vacuum speed of light
cnmfs = SI.c*1e-15*1e9 # nm/fs, vacuum speed of light
heVfs = SI.codata.value('Planck constant in eV s')*1e15 # eV*fs, Planck's constant
hbareVfs = heVfs/(2*pi) # eV*fs, h-bar, reduced Planck's constant
eMass0eVnmfs = SI.codata.value('electron mass energy equivalent in MeV')*1e6/(cnmfs*cnmfs) # eV/c^2 (c in nm/fs) = eV*fs^2/nm^2, electron mass
eps0enmV = 1/(2*SI.alpha*heVfs*cnmfs) # e^2/(eV*nm), permittivity of free space


# ======== Function definitions ========
# -------- Check to see if functions are loaded --------
def printCheck():
    print("  ... Mie functions loaded")
    return

# -------- Basis functions --------
# Returns list of Riccati-Bessel functions and derivatives 
# at shell interface radius R
# returns all orders up to lmax
def riccatiBessels(lmax, n1, n2, R, k0):
    lmax = int(lmax)
    k1 = n1*k0 # k in medium 1
    k2 = n2*k0 # k in medium 2
    
    # functions in medium 1
    (j1, dj1, y1, dy1) = sph_jnyn(lmax, k1*R +1j*1e-9) # depricated in scipy 0.18?
    # prepare for new SciPy bessel function definitions
    # j1 = spherical_jn(lmax, k1*R)
    # dj1 = spherical_jn(lmax, k1*R, derivative=True)
    # y1 = spherical_yn(lmax, k1*R)
    # dy1 = spherical_yn(lmax, k1*R, derivative=True) 
    h1 = j1 + y1*1j
    dh1 = dj1 + dy1*1j
    u1 = R*j1
    du1 = j1 + k1*R*dj1
    w1 = R*h1
    dw1 = h1 + k1*R*dh1
    # functions in medium 2
    (j2, dj2, y2, dy2) = sph_jnyn(lmax, k2*R +1j*1e-9) # depricated in scipy 0.18?
    # prepare for new SciPy bessel function definitions
    # j2 = spherical_jn(lmax, k2*R)
    # dj2 = spherical_jn(lmax, k2*R, derivative=True)
    # y2 = spherical_yn(lmax, k2*R)
    # dy2 = spherical_yn(lmax, k2*R, derivative=True)
    h2 = j2 + y2*1j
    dh2 = dj2 + dy2*1j
    u2 = R*j2
    du2 = j2 + k2*R*dj2
    w2 = R*h2
    dw2 = h2 + k2*R*dh2
    
    return (u1, du1, w1, dw1, u2, du2, w2, dw2)


# -------- Transfer matrices --------
# Returns transfer matrices up to order lmax
# 4 types: (Backward 12,Forward 21) x (TM,TE)
# (A1,B1) = T12 (A2,B2) ==> Backward
# (A2,B2) = T21 (A1,B2) ==> Forward
# 1a) Backward 12 TM transfer matrix list
# Transfer matrix of order l
def t12m(n1, n2, k0, u1, du1, w1, dw1, u2, du2, w2, dw2):
    k1 = n1*k0 # k in medium 1
    k2 = n2*k0 # k in medium 2
    t0 = [[dw1*u2 - w1*du2, dw1*w2 - w1*dw2],
          [-du1*u2 + u1*du2, -du1*w2 + u1*dw2]]
    return np.multiply(-k1*1j,t0)
def T12MList(lmax, n1, n2, R, k0):
    # Compute lists of basis functions up to order lmax
    (u1, du1, w1, dw1, u2, du2, w2, dw2) = riccatiBessels(lmax, n1, n2, R, k0)
    # Compute list of matrices up to order lmax
    TList = [] # initialize
    for l in range(0, lmax+1):
        TList.append(t12m(n1, n2, k0, u1[l], du1[l], w1[l], dw1[l], u2[l], du2[l], w2[l], dw2[l]))
    return TList

# 1b) Backward 12 TE transfer matrix list
# Single transfer matrix of order l
def t12e(n1, n2, k0, u1, du1, w1, dw1, u2, du2, w2, dw2):
    k1 = n1*k0 # k in medium 1
    k2 = n2*k0 # k in medium 2
    t0 = [[dw1*u2*n2/n1 - w1*du2*n1/n2, dw1*w2*n2/n1 - w1*dw2*n1/n2],
          [-du1*u2*n2/n1 + u1*du2*n1/n2, -du1*w2*n2/n1 + u1*dw2*n1/n2]]
    return np.multiply(-k1*1j,t0)
def T12EList(lmax, n1, n2, R, k0):
    # Compute lists of basis functions up to order lmax
    (u1, du1, w1, dw1, u2, du2, w2, dw2) = riccatiBessels(lmax, n1, n2, R, k0)
    # Compute list of matrices up to order lmax
    TList = [] # initialize
    for l in range(0, lmax+1):
        TList.append(t12e(n1, n2, k0, u1[l], du1[l], w1[l], dw1[l], u2[l], du2[l], w2[l], dw2[l]))
    return TList

# 2a) Forward 21 TM transfer matrix list
# Single transfer matrix of order l
def t21m(n1, n2, k0, u1, du1, w1, dw1, u2, du2, w2, dw2):
    k1 = n1*k0 # k in medium 1
    k2 = n2*k0 # k in medium 2
    t0 = [[dw2*u1 - w2*du1, dw2*w1 - w2*dw1],
          [-du2*u1 + u2*du1, -du2*w1 + u2*dw1]]
    return np.multiply(-k2*1j,t0)
def T21MList(lmax, n1, n2, R, k0):
    # Compute lists of basis functions up to order lmax
    (u1, du1, w1, dw1, u2, du2, w2, dw2) = riccatiBessels(lmax, n1, n2, R, k0)
    # Compute list of matrices up to order lmax
    TList = [] # initialize
    for l in range(0, lmax+1):
        TList.append(t21m(n1, n2, k0, u1[l], du1[l], w1[l], dw1[l], u2[l], du2[l], w2[l], dw2[l]))
    return TList

# 2b) Forward 21 TE transfer matrix list
# Single transfer matrix of order l
def t21e(n1, n2, k0, u1, du1, w1, dw1, u2, du2, w2, dw2):
    k1 = n1*k0 # k in medium 1
    k2 = n2*k0 # k in medium 2
    t0 = [[dw2*u1*n1/n2 - w2*du1*n2/n1, dw2*w1*n1/n2 - w2*dw1*n2/n1],
        [-du2*u1*n1/n2 + u2*du1*n2/n1, -du2*w1*n1/n2 + u2*dw1*n2/n1]]
    return np.multiply(-k2*1j,t0)
def T21EList(lmax, n1, n2, R, k0):
    # Compute lists of basis functions up to order lmax
    (u1, du1, w1, dw1, u2, du2, w2, dw2) = riccatiBessels(lmax, n1, n2, R, k0)
    # Compute list of matrices up to order lmax
    TList = [] # initialize
    for l in range(0, lmax+1):
        TList.append(t21e(n1, n2, k0, u1[l], du1[l], w1[l], dw1[l], u2[l], du2[l], w2[l], dw2[l]))
    return TList


# -------- Ordered product matrices --------
# Returns list of ordered product transfer matrices up to order lmax
# (i,j) = 0...(N-1) for N refractive indexes and N-1 shell radii
# If i < j, returns backward product matrix, 
#  e.g.: T0j = T01*T12*... so that (A0,B0) = T0j*(Aj,Bj)
# If i > j, returns forward product matrix, 
#  e.g.: Ti0 = ...*T21*T10 so that (Ai,Bi) = Ti0*(A0,B0)
# a) TM-polarized
def TijMList(lmax, ns, Rs, k0, i, j):
    TLists = []
    TijList = []
    if i < 0 or j < 0 or i >= len(ns) or j >= len(ns): return [[0,0],[0,0]]
    elif i == j: return [[1,0],[0,1]]
    elif i < j: # Backward transfer
        for iRI in range(i, j): # Compute all transfer matrices lmax x |i-j|
            TList = T12MList(lmax, ns[iRI], ns[iRI+1], Rs[iRI], k0)
            TLists.append(TList)
        for l in range(0, lmax+1):
            Tij = TLists[0][l]
            for iRI in range(1, j-i): # Construct ordered products
                T12 = TLists[iRI][l]
                Tij = np.dot(Tij, T12)
            TijList.append(Tij)
        return TijList
    elif i > j: # Forward transfer
        for iRI in range(j, i): # Compute all transfer matrices lmax x |i-j|
            TList = T21MList(lmax, ns[iRI], ns[iRI+1], Rs[iRI], k0)
            TLists.append(TList)
        for l in range(0, lmax+1): # Loop over order l
            Tij = TLists[0][l]
            for iRI in range(1, i-j): # Construct ordered products
                T21 = TLists[iRI][l]
                Tij = np.dot(T21, Tij)
            TijList.append(Tij)
        return TijList
    else: return [[0,0],[0,0]]
# b) TE-polarized
def TijEList(lmax, ns, Rs, k0, i, j):
    TLists = []
    TijList = []
    if i < 0 or j < 0 or i >= len(ns) or j >= len(ns): return [[0,0],[0,0]]
    elif i == j: return [[1,0],[0,1]]
    elif i < j: # Backward transfer
        for iRI in range(i, j): # Compute all transfer matrices lmax x |i-j|
            TList = T12EList(lmax, ns[iRI], ns[iRI+1], Rs[iRI], k0)
            TLists.append(TList)
        for l in range(0, lmax+1):
            Tij = TLists[0][l]
            for iRI in range(1, j-i): # Construct ordered products
                T12 = TLists[iRI][l]
                Tij = np.dot(Tij, T12)
            TijList.append(Tij)
        return TijList
    elif i > j: # Forward transfer
        for iRI in range(j, i): # Compute all transfer matrices lmax x |i-j|
            TList = T21EList(lmax, ns[iRI], ns[iRI+1], Rs[iRI], k0)
            TLists.append(TList)
        for l in range(0, lmax+1): # Loop over order l
            Tij = TLists[0][l]
            for iRI in range(1, i-j): # Construct ordered products
                T21 = TLists[iRI][l]
                Tij = np.dot(T21, Tij)
            TijList.append(Tij)
        return TijList
    else: return [[0,0],[0,0]]


# -------- Scattering coefficients --------
# Returns list of scattering coefficients up to order lmax
# TM-polarized
def bTMList(lmax, ns, Rs, k0):
    TList = TijMList(lmax, ns, Rs, k0, np.size(Rs), 0)
    cList = []
    for l in range(0,lmax+1):
        cList.append(-TList[l][1,0]/TList[l][0,0])
    return cList
# TE-polarized
def aTEList(lmax, ns, Rs, k0):
    TList = TijEList(lmax, ns, Rs, k0, np.size(Rs), 0)
    cList = []
    for l in range(0,lmax+1):
        cList.append(-TList[l][1,0]/TList[l][0,0])
    return cList


# -------- Optical cross sections --------
# Scattering cross sections
# TM list of all orders
def CscaMList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    bList = bTMList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        bl = abs(bList[l])
        CList.append(coeffl*bl*bl)
    return CList
# TE list of all orders
def CscaEList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    aList = aTEList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        al = abs(aList[l])
        CList.append(coeffl*al*al)
    return CList
# Total (sum over l and polarization)
def Csca(lmax, ns, Rs, k0):
    CM = CscaMList(lmax, ns, Rs, k0)
    CE = CscaEList(lmax, ns, Rs, k0)
    Ctot = 0
    for l in range(0,lmax):
        Ctot = Ctot + CM[l] + CE[l]
    return Ctot

# Absorption cross sections
# TM list of all orders
def CabsMList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    bList = bTMList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        bl = abs(-bList[l]+1)
        CList.append(coeffl*(1-bl*bl))
    return CList
# TE list of all orders
def CabsEList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    aList = aTEList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        al = abs(-aList[l]+1)
        CList.append(coeffl*(1-al*al))
    return CList
# Total (sum over l and polarization)
def Cabs(lmax, ns, Rs, k0):
    CM = CabsMList(lmax, ns, Rs, k0)
    CE = CabsEList(lmax, ns, Rs, k0)
    Ctot = 0
    for l in range(0,lmax):
        Ctot = Ctot + CM[l] + CE[l]
    return Ctot

# Extinction cross sections
# TM list of all orders
def CextMList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    bList = bTMList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        bl = 2*np.real(bList[l])
        CList.append(coeffl*bl)
    return CList
# TE list of all orders
def CextEList(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    aList = aTEList(lmax, ns, Rs, k0)
    CList = []
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        al = 2*np.real(aList[l])
        CList.append(coeffl*al)
    return CList
# Total (sum over l and polarization)
def Cext(lmax, ns, Rs, k0):
    CM = CextMList(lmax, ns, Rs, k0)
    CE = CextEList(lmax, ns, Rs, k0)
    Ctot = 0
    for l in range(0,lmax):
        Ctot = Ctot + CM[l] + CE[l]
    return Ctot

# Total extinction, absorption, scattering
def CExtAbsSca(lmax, ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    aList = aTEList(lmax, ns, Rs, k0)
    bList = bTMList(lmax, ns, Rs, k0)
    CExt, CAbs, CSca = 0, 0, 0
    for l in range(1, lmax+1):
        coeffl = (2*pi/(k*k))*((2*l+1)/(l*(l+1)))
        al = 2*np.real(aList[l])
        bl = 2*np.real(bList[l])
        CExt = CExt + coeffl*(al+bl)
        al = abs(-aList[l]+1)
        bl = abs(-bList[l]+1)
        CAbs = CAbs + coeffl*(2-al*al-bl*bl)
        al = abs(aList[l])
        bl = abs(bList[l])
        CSca = CSca + coeffl*(al*al+bl*bl)
    return [CExt, CAbs, CSca]

# Efficiency
def Qeff(R,C): return C/(2*pi*R*R)

# Dipole polarizability
def alpha(ns, Rs, k0):
    nmed = np.real(ns[-1])
    k = nmed*k0
    a = aTEList(1, ns, Rs, k0)[0]
    return 1j*(3/2)*a/(k*k*k)
