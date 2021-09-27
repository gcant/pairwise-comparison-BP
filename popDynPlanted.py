import ctypes
import numpy as np

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/PDP.so")

lib.initPopulation.argtypes = [ ctypes.c_int, ctypes.c_int ]
lib.initPopulation.restype = None

lib.PoissonPopDyn.argtypes = [ ctypes.c_double, ctypes.c_int,
        ctypes.c_int, array_double]
lib.PoissonPopDyn.restype = None

def initPopulation(N, L):
    lib.initPopulation( ctypes.c_int(N), ctypes.c_int(L) )

def PopulationBurnIn(c, n):
    out = np.zeros(n)
    lib.PoissonPopDyn(ctypes.c_double(c), ctypes.c_int(n), ctypes.c_int(0), out)

def iteratePopulation(c, n, entropy_samples=2000):
    out = np.zeros(n)
    lib.PoissonPopDyn( ctypes.c_double(c), ctypes.c_int(n), ctypes.c_int(entropy_samples), out )
    return out


if __name__=="__main__":
    N = 1000
    L = 64
    initPopulation(N,L)
    A = []
    for c in np.linspace(0,4,11):
        print(c)
        PopulationBurnIn(c,1000)
        A.append( iteratePopulation(c, 10) )
    print( np.array(A) )
