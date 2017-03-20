# ======== Header for ipyparallel kernels  ========
import os, sys, types
import ipyparallel as ipp


# -------- Parallel kernels --------
print("Initializing cluster ...")

# variables
global kernels, cluster, nKernels
kernels = ipp.Client()
print("   Client variable \'kernels\'")
cluster = kernels[:]
print("   Cluster Direct View variable \'cluster\'")
nKernels = len(kernels.ids)
print("   Variable \'nKernels\' =",nKernels)

# change cluster current working directory
def f(cwd): os.chdir(cwd); print(os.getcwd); return
cwd = os.getcwd()
cwdList = []
for i in range(nKernels): cwdList.append(cwd)
with cluster.sync_imports(): import os
cluster.map_sync(f, cwdList)

# initialize parallel kernels

#with cluster.sync_imports():
#    import mie
#    import refractive_index_library as ri
