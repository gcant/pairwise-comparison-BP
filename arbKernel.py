import ctypes
import numpy as np
import networkx as nx

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/arbKernel.so")

lib.full_algorithm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        array_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        array_double, array_double,ctypes.c_int]
lib.full_algorithm.restype = ctypes.c_int 

lib.compute_Cheb_pts.argtypes = [ array_double, ctypes.c_int ]
lib.compute_Cheb_pts.restype = None

def ChebPts(L):
    ans = np.zeros(L)
    lib.compute_Cheb_pts(ans,L)
    return ans

def BP(G, L, beta, tol=10e-10,sm=0):
    M = np.array(list(G.edges())).flatten().astype(ctypes.c_int)
    output = np.zeros(L*(G.number_of_nodes()+1) + 2)
    edgescores = np.zeros(G.number_of_edges())

    s = lib.full_algorithm(
        ctypes.c_int(G.number_of_nodes()),
        ctypes.c_int(G.number_of_edges()),
        M,
        ctypes.c_int(L),
        ctypes.c_double(beta),
        ctypes.c_double(tol),
        output, edgescores,ctypes.c_int(sm))

    pts = output[:L].copy()
    A = output[L:-2].reshape(G.number_of_nodes(),L)
    S = output[-2]
    lnZ = output[-1]
    return pts, A, S, lnZ 

