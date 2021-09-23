import ctypes
import numpy as np
import networkx as nx

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./LEMP.so")

lib.full_algorithm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        array_int,
        ctypes.c_int,
        ctypes.c_double,
        array_double]
lib.full_algorithm.restype = ctypes.c_int 


def BP(G, L, tol=10e-10):
    M = np.array(list(G.edges())).flatten().astype(ctypes.c_int)
    output = np.zeros(L*(G.number_of_nodes()+1) + 4)

    s = lib.full_algorithm(
        ctypes.c_int(G.number_of_nodes()),
        ctypes.c_int(G.number_of_edges()),
        M,
        ctypes.c_int(L),
        ctypes.c_double(tol),
        output)

    pts = output[:L].copy()
    A = output[L:-4].reshape(G.number_of_nodes(),L)
    S1 = output[-4]/G.number_of_nodes()
    S2 = output[-3]/G.number_of_nodes()
    S = output[-2]
    logV = output[-1]
    density_states = S - G.number_of_nodes()*np.log(2)
    return pts, A, S1, S2, density_states, S, logV, s

