## ======== Refractive index library ======== ##
# load refractive index from data and convert to functions of wavelength (nm)
# includes confinement effects for metals
# includes semi-analytic models for many solvents

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

# Scientific python for interpolations
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


# -------- Check to see if functions are loaded --------
def printCheck():
    print("  ... refractive index functions loaded:")
    print("       Au, Ag, SiO2, ITO, ZnO, H2O, ")
    print("       ethanol, CS2, benzene, toluene, ")
    print("       chloroform, methanol, butanol, ")
    print("       propanol")
    return


# -------- Metals --------

# Drude model metal
def nDrude(EPeV, GeV, epsInf, wlnm):
    EeV = heVfs*cnmfs/wlnm
    eps = epsInf - EPeV**2/(EeV**2 + 1j*GeV*EeV)
    n = sqrt(eps)
    return n

## Au ##

# Au default: [188,1937] nm
# Johnson & Christy (1972)
# dx.doi.org/10.1103/PhysRevB.6.4370
nAuArray = np.genfromtxt("optical-data/Johnson-Au.csv", delimiter=",", skip_header=0)
# Interpolation function
nAuRe = interpolate.interp1d(1e3*nAuArray[:,0], nAuArray[:,1], kind="cubic")
nAuIm = interpolate.interp1d(1e3*nAuArray[:,0], nAuArray[:,2], kind="cubic")
def nAu(wl0nm):
    return nAuRe(wl0nm) + 1j*nAuIm(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# Formulas from Bohren & Huffman, Ch. 12
# For spheres, lsc = (4/3)*Radius
# For shells, lsc = thickness
def nAuSc(wl0nm, lscnm): 
    nMetal = nAu(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.40 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 9.06 # eV, bulk plasmon energy (calculated from m*)
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff
# more general scattering law
# first order correction +i*(E1eV/EeV)^3
# second order correction + (E2eV/EeV)^4
# with E1^3 = (Ep^2*hb*vF/lsc); E1 < 2.3 eV for r > 5 nm
# and E2^2 = (Ep*hb*vF/lsc); E2 < 1.2 eV for r > 5 nm
def nAuScGen(wl0nm, E1eV, E2eV): 
    nMetal = nAu(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    epsEff = epsMetal + 1j*(E1eV/EeV)**3 + (E2eV/EeV)**4
    neff = sqrt(epsEff)
    return neff

# Au: [300,2493] nm
# Olmon (2012) - template-stripped (lowest uncertainty)
# dx.doi.org/10.1103/PhysRevB.86.235147
# Bulk plasmon energy = 8.8 +/-0.05 eV
# relaxation rate Gamma = 13 +/-1 fs
nAu2012tsArray = np.genfromtxt("optical-data/Olmon_PRB2012_TS.dat", delimiter="\t", skip_header=2)
# Interpolation function
nAu2012tsRe = interpolate.interp1d(1e9*nAu2012tsArray[:,1], nAu2012tsArray[:,-2], kind="cubic")
nAu2012tsIm = interpolate.interp1d(1e9*nAu2012tsArray[:,1], nAu2012tsArray[:,-1], kind="cubic")
def nAu2012ts(wl0nm):
    return nAu2012tsRe(wl0nm) + 1j*nAu2012tsIm(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# Formulas derived following Bohren & Huffman, Ch. 12
# For spheres, lsc = (4/3)*radius
# For shells, lsc = thickness
def nAu2012tsSc(wl0nm, lscnm): 
    nMetal = nAu2012ts(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.40 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 8.8 # eV, bulk plasmon energy
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff
# more general scattering law
# first order correction +i*(E1eV/EeV)^3
# second order correction + (E2eV/EeV)^4
# with E1^3 = (Ep^2*hb*vF/lsc); E1 < 5 eV for r > 2 nm
# and E2^2 = (Ep*hb*vF/lsc); E2 < 2.5 eV for r > 2 nm
def nAu2012tsScGen(wl0nm, E1eV, E2eV): 
    nMetal = nAu2012ts(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    epsEff = epsMetal + 1j*(E1eV/EeV)**3 + (E2eV/EeV)**4
    neff = sqrt(epsEff)
    return neff

# Au: [300,2493] nm
# Olmon (2012) - single-crystal
# dx.doi.org/10.1103/PhysRevB.86.235147
# Bulk plasmon energy = 8.1 +/-0.8 eV
# relaxation rate Gamma = 14 +/-4 fs
nAu2012scArray = np.genfromtxt("optical-data/Olmon_PRB2012_SC.dat", delimiter="\t", skip_header=2)
# Interpolation function
nAu2012scRe = interpolate.interp1d(1e9*nAu2012scArray[:,1], nAu2012scArray[:,-2], kind="cubic")
nAu2012scIm = interpolate.interp1d(1e9*nAu2012scArray[:,1], nAu2012scArray[:,-1], kind="cubic")
def nAu2012sc(wl0nm):
    return nAu2012scRe(wl0nm) + 1j*nAu2012scIm(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# Formulas derived following Bohren & Huffman, Ch. 12
# For spheres, lsc = (4/3)*radius
# For shells, lsc = thickness
def nAu2012scSc(wl0nm, lscnm): 
    nMetal = nAu2012sc(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.40 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 8.1 # eV, bulk plasmon energy
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff
# more general scattering law
# first order correction +i*(E1eV/EeV)^3
# second order correction + (E2eV/EeV)^4
# with E1^3 = (Ep^2*hb*vF/lsc); E1 < 5 eV for r > 2 nm
# and E2^2 = (Ep*hb*vF/lsc); E2 < 2.5 eV for r > 2 nm
def nAu2012scScGen(wl0nm, E1eV, E2eV): 
    nMetal = nAu2012sc(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    epsEff = epsMetal + 1j*(E1eV/EeV)**3 + (E2eV/EeV)**4
    neff = sqrt(epsEff)
    return neff

# Au: [300,2493] nm
# Olmon (2012) - evaporated film
# dx.doi.org/10.1103/PhysRevB.86.235147
# Bulk plasmon energy = 8.5 +/-0.5 eV
# relaxation rate Gamma = 14 +/-3 fs
nAu2012evArray = np.genfromtxt("optical-data/Olmon_PRB2012_EV.dat", delimiter="\t", skip_header=2)
# Interpolation function
nAu2012evRe = interpolate.interp1d(1e9*nAu2012evArray[:,1], nAu2012evArray[:,-2], kind="cubic")
nAu2012evIm = interpolate.interp1d(1e9*nAu2012evArray[:,1], nAu2012evArray[:,-1], kind="cubic")
def nAu2012ev(wl0nm):
    return nAu2012evRe(wl0nm) + 1j*nAu2012evIm(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# Formulas derived following Bohren & Huffman, Ch. 12
# For spheres, lsc = (4/3)*radius
# For shells, lsc = thickness
def nAu2012evSc(wl0nm, lscnm): 
    nMetal = nAu2012ev(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.40 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 8.5 # eV, bulk plasmon energy
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff
# more general scattering law
# first order correction +i*(E1eV/EeV)^3
# second order correction + (E2eV/EeV)^4
# with E1^3 = (Ep^2*hb*vF/lsc); E1 < 5 eV for r > 2 nm
# and E2^2 = (Ep*hb*vF/lsc); E2 < 2.5 eV for r > 2 nm
def nAu2012evScGen(wl0nm, E1eV, E2eV): 
    nMetal = nAu2012ev(wl0nm) # refractive index of Au
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    epsEff = epsMetal + 1j*(E1eV/EeV)**3 + (E2eV/EeV)**4
    neff = sqrt(epsEff)
    return neff


## Ag ##

# Ag default: [188,1937] nm
# Johnson & Christy (1972)
# dx.doi.org/10.1103/PhysRevB.6.4370
nAgArray = np.genfromtxt("optical-data/Johnson-Ag.csv", delimiter=",", skip_header=0)
# Interpolation function
nAgRe = interpolate.interp1d(1e3*nAgArray[:,0], nAgArray[:,-2], kind="cubic")
nAgIm = interpolate.interp1d(1e3*nAgArray[:,0], nAgArray[:,-1], kind="cubic")
def nAg(wl0nm):
    return nAgRe(wl0nm) + 1j*nAgIm(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# Formulas from Bohren & Huffman, Ch. 12
# Data from J&C (some calculated)
# For spheres, lsc = (4/3)*Radius
# For shells, lsc = thickness
def nAgSc(wl0nm, lscnm): 
    nMetal = nAg(wl0nm) # refractive index of Ag
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.39 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 9.17 # eV, bulk plasmon energy (calculated from m*)
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff
# more general scattering law
# first order correction +i*(E1eV/EeV)^3
# second order correction + (E2eV/EeV)^4
# with E1^3 = (Ep^2*hb*vF/lsc); E1 < 2.3 eV for r > 5 nm
# and E2^2 = (Ep*hb*vF/lsc); E2 < 1.2 eV for r > 5 nm
def nAgScGen(wl0nm, E1eV, E2eV): 
    nMetal = nAg(wl0nm) # refractive index of Ag
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    epsEff = epsMetal + 1j*(E1eV/EeV)**3 + (E2eV/EeV)**4
    neff = sqrt(epsEff)
    return neff

# Ag: [300,2000] nm
# Jiang, Pillai, Green (2016)
# dx.doi.org/10.1038/srep30605
nAg2016Array = np.genfromtxt("optical-data/unsw-Ag.csv", delimiter=",", skip_header=1)
# Interpolation function
nAg2016Re = interpolate.interp1d(nAg2016Array[:,0], nAg2016Array[:,1], kind="cubic")
nAg2016Im = interpolate.interp1d(nAg2016Array[:,0], nAg2016Array[:,2], kind="cubic")
def nAg2016(wl0nm):
    return nAg2016Re(wl0nm) + 1j*nAg2016Im(wl0nm)
# Include electron scattering for confined geometries (thin shells, small particles)
# From Bohren & Huffman, Ch. 12
# For spheres, lsc = (4/3)*Radius
# For shells, lsc = thickness
def nAg2016Sc (wl0nm, lscnm): 
    nMetal = nAg2016(wl0nm) # refractive index of Ag
    epsMetal = nMetal**2 # relative permittivity
    EeV = heVfs*cnmfs/wl0nm # rad/fs, light angular frequency
    vfnmfs = 1.39 # nm/fs, mean electron speed, Ashcroft & Mermin
    EPeV = 8.74 # eV, bulk plasmon energy, fitted from eps1 vs wl^2, wl:[1200,2000] nm
    epsEff = epsMetal + 1j*hbareVfs*( (vfnmfs)/(lscnm) )*( (EPeV**2)/(EeV**3) )
    neff = sqrt(epsEff)
    return neff



# -------- Solid dieletrics (substrates) --------

# Silica [252,1250] nm
# Gao (2013)
nSiO2Array = np.genfromtxt("optical-data/SiO2-Gao-nk.txt", delimiter="\t")
# Interpolation function
nSiO2Re = interpolate.interp1d(1000*nSiO2Array[:,0], nSiO2Array[:,1], kind="cubic")
nSiO2Im = interpolate.interp1d(1000*nSiO2Array[:,0], nSiO2Array[:,2], kind="cubic")
def nSiO2 (wl0nm):
    return nSiO2Re(wl0nm) + 1j*nSiO2Im(wl0nm)

# ITO [252,1000] nm
# Koenig (2014)
nITOArray = np.genfromtxt("optical-data/ITO.txt", delimiter="\t")
# Interpolation function
nITORe = interpolate.interp1d(1000*nITOArray[:,0], nITOArray[:,1], kind="cubic")
nITOIm = interpolate.interp1d(1000*nITOArray[:,0], nITOArray[:,2], kind="cubic")
def nITO (wl0nm):
    return nITORe(wl0nm) + 1j*nITOIm(wl0nm)

# ZnO [450,800] nm
# Hu 1997
def nZnO (wl0nm):
    A1 = 1.9281
    A2 = -1.1157e-5
    A3 = 5.9696e-3
    wlum = wl0nm*1e-3
    return A1 + A2/(wlum*wlum) + A3/(pow(wlum,4))


# -------- Liquids & solvents --------

# Water [200,1100] nm
# Daimon 2007
def nH2O (wl0nm):
    A1 = 5.684027565e-1
    A2 = 1.726177391e-1
    A3 = 2.086189578e-2
    A4 = 1.130748688e-1
    wlum = wl0nm*1e-3
    wl1um2 = 5.101829712e-3
    wl2um2 = 1.821153936e-2
    wl3um2 = 2.620722293e-2
    wl4um2 = 1.06979271e1
    return sqrt(1 + 
                A1*(wlum*wlum)/(wlum*wlum-wl1um2) + 
                A2*(wlum*wlum)/(wlum*wlum-wl2um2) + 
                A3*(wlum*wlum)/(wlum*wlum-wl3um2) + 
                A4*(wlum*wlum)/(wlum*wlum-wl4um2)
                )

# Ethanol [500,1600] nm
# Kedenburg (2012)
def nEtOH (wl0nm):
    C0 = 1.83347
    C1 = 0.00648
    C2 = 0.00031
    C3 = 0.
    C4 = 0.
    C5 = -0.00352
    wlum = wl0nm*1e-3
    return sqrt(C0 + 
                C1/pow(wlum,2) + 
                C2/pow(wlum,4) + 
                C3/pow(wlum,6) + 
                C4/pow(wlum,8) + 
                C5*pow(wlum,2)
               )


# Carbon disulfide [300,2500] nm
# Samoc 2003
def nCS2 (wl0nm):
    C0 = 1.582445
    C1 = 13.7372e3
    C2 = 10.0243e8
    C3 = -15.6572e13
    C4 = 1.8294e18
    C5 = -3.2117e-10
    return sqrt(C0 + 
                C1/pow(wl0nm,2) + 
                C2/pow(wl0nm,4) + 
                C3/pow(wl0nm,6) + 
                C4/pow(wl0nm,8) + 
                C5*pow(wl0nm,2)
               )

# Benzene [300,2500] nm
# Samoc 2003
def nBenz (wl0nm):
    C0 = 1.473644
    C1 = 11.26920e3
    C2 = -9.2034e8
    C3 = 12.4302e13
    C4 = -3.9224e18
    C5 = 4.8623e-10
    return sqrt(C0 + 
                C1/pow(wl0nm,2) + 
                C2/pow(wl0nm,4) + 
                C3/pow(wl0nm,6) + 
                C4/pow(wl0nm,8) + 
                C5*pow(wl0nm,2)
               )

# Toluene [300,2500] nm
# Samoc 2003
def nTolu (wl0nm):
    C0 = 1.474775
    C1 = 6.99031e3
    C2 = 2.1776e8
    C3 = 0.
    C4 = 0.
    C5 = 0.
    return sqrt(C0 + 
                C1/pow(wl0nm,2) + 
                C2/pow(wl0nm,4) + 
                C3/pow(wl0nm,6) + 
                C4/pow(wl0nm,8) + 
                C5*pow(wl0nm,2)
               )

# Chloroform [300,2500] nm
# Samoc 2003
def nChloro (wl0nm):
    C0 = 1.431364
    C1 = 5.63241e3
    C2 = -2.0805e8
    C3 = 1.2613e13
    C4 = 0.
    C5 = 0.
    return sqrt(C0 + 
                C1/pow(wl0nm,2) + 
                C2/pow(wl0nm,4) + 
                C3/pow(wl0nm,6) + 
                C4/pow(wl0nm,8) + 
                C5*pow(wl0nm,2)
               )

# Methanol [400,1600] nm
# Moutzouris 20133
def nMeOH (wl0nm):
    wlum = 1e-3*wl0nm
    A0 = 1.745946239
    A1 = -0.005362181
    A2 = 0.004656355
    A3 = 0.0044714
    A4 = -0.000015087
    return sqrt(A0 + 
                A1*pow(wlum,2) + 
                A2/pow(wlum,2) + 
                A3/pow(wlum,4) + 
                A4/pow(wlum,6)
               )

# Butanol (1) [400,1600] nm
# Moutzouris 20133
def nBuOH (wl0nm):
    wlum = 1e-3*wl0nm
    A0 = 1.917816501
    A1 = -0.00115077
    A2 = 0.01373734
    A3 = -0.00194084
    A4 = 0.000254077
    return sqrt(A0 + 
                A1*pow(wlum,2) + 
                A2/pow(wlum,2) + 
                A3/pow(wlum,4) + 
                A4/pow(wlum,6)
               )

# Propanol (1) [400,1600] nm
# Moutzouris 20133
def nPrOH (wl0nm):
    wlum = 1e-3*wl0nm
    A0 = 1.89400242
    A1 = -0.003349425
    A2 = 0.004418653
    A3 = 0.00108023
    A4 = -0.000067337
    return sqrt(A0 + 
                A1*pow(wlum,2) + 
                A2/pow(wlum,2) + 
                A3/pow(wlum,4) + 
                A4/pow(wlum,6)
               )

