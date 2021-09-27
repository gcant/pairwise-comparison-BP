import ctypes
import numpy as np
import networkx as nx

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/fixedCost.so")

lib.full_algorithm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        array_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        array_double, array_double,ctypes.c_int]
lib.full_algorithm.restype = ctypes.c_int 

lib.set_prior.argtypes = [ array_double, ctypes.c_int ]
lib.set_prior.restype = None

lib.compute_Cheb_pts.argtypes = [ array_double, ctypes.c_int ]
lib.compute_Cheb_pts.restype = None

def ChebPts(L):
    ans = np.zeros(L)
    lib.compute_Cheb_pts(ans,L)
    return ans

def log0(x):
    if x<=0.:
        return 0.
    else:
        return np.log(x)

def BP(G, L, Delta, tol=10e-10, prior= lambda x: 0.*x+0.5,sm=0):
    M = np.array(list(G.edges())).flatten().astype(ctypes.c_int)
    output = np.zeros(L*(G.number_of_nodes()+1) + 6)
    edgescores = np.zeros(G.number_of_edges())

    if prior:
        lib.set_prior( prior(ChebPts(L)), L )

    s = lib.full_algorithm(
        ctypes.c_int(G.number_of_nodes()),
        ctypes.c_int(G.number_of_edges()),
        M,
        ctypes.c_int(L),
        ctypes.c_double(Delta),
        ctypes.c_double(tol),
        output, edgescores,ctypes.c_int(sm))

    pts = output[:L].copy()
    A = output[L:-6].reshape(G.number_of_nodes(),L)

    UP = output[-6]

    S1 = output[-5]/G.number_of_nodes()
    S2 = output[-4]/G.number_of_nodes()
    S = output[-3]
    logV = output[-2]
    EV = output[-1]
    density_states = S - G.number_of_nodes()*np.log(2)
    logV = S + log0(Delta)*EV + UP
    return pts, A, UP, S1, S2, density_states, S, logV, EV, s, edgescores, M

def random_gnp(n, p, s=0.5):
    G = nx.fast_gnp_random_graph(n,p)
    DG = nx.DiGraph()
    for i in range(n):
        DG.add_node(i)
    for i,j in G.edges():
        i,j = min(i,j),max(i,j)
        if np.random.random()<(1-s):
            DG.add_edge(i,j)
        else:
            DG.add_edge(j,i)
    return DG

if __name__=="__main__":
    n = 5000
    G = random_gnp(n, 1.0/n)
    print( BP(G,64,0.01)[-3]/n )

