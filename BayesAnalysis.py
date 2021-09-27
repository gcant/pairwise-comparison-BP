import arbKernel
import fixedCost
import numpy as np
import networkx as nx
import scipy.integrate

def freeEnergyBTL(G, L, beta, verbose=False):
    lnZ = arbKernel.BP(G,L,beta)[3]
    n = G.number_of_nodes()
    if verbose: print(beta,lnZ-n*np.log(2.))
    return lnZ-n*np.log(2.)

def freeEnergyDelta(G, L, Delta, verbose=False):
    lnZ = fixedCost.BP(G,L,Delta)[7]
    m = G.number_of_edges()
    if verbose: print(Delta, lnZ-m*np.log(1+Delta))
    return lnZ-m*np.log(1+Delta)

def generate_Delta(G_undirected, Delta):
    n = G_undirected.number_of_nodes()
    G = nx.DiGraph()
    x = np.random.permutation(np.arange(n))
    win_prob = 1./(Delta+1.)
    for i in G_undirected.nodes():
        G.add_node(i)
    for i,j in G_undirected.edges():
        ii,jj = max(x[i],x[j]),min(x[i],x[j])
        if np.random.random()<win_prob:
            G.add_edge(ii,jj)
        else:
            G.add_edge(jj,ii)
    return G

def generate_BTL(G_undirected, beta):
    n = G_undirected.number_of_nodes()
    G = nx.DiGraph()
    x = np.random.random(size=n)*2-1.
    for i in G_undirected.nodes():
        G.add_node(i)
    for i,j in G_undirected.edges():
        win_prob = 1./(1.+np.exp(beta*(x[i]-x[j])))
        if np.random.random()<win_prob:
            G.add_edge(i,j)
        else:
            G.add_edge(j,i)
    return G

def integrated_Likelihood(X,ans):
    prefactor = np.max(ans)
    P = np.exp(ans-np.max(ans))
    return prefactor + np.log(np.abs(scipy.integrate.simps(P,X)))

if __name__=="__main__":
    #G = generate_Delta(nx.gnp_random_graph(1000,4./999), 0.3)
    G = generate_BTL(nx.gnp_random_graph(1000,4./999), 1.6)

    Y = np.linspace(0.01,0.99,32)
    BTL_posterior = np.array([ freeEnergyBTL(G,32,y,verbose=True) for y in -np.log(Y) ])
    Delta_posterior = np.array([ freeEnergyDelta(G,32,y,verbose=True) for y in Y ])
  
    BTL_posterior[np.isnan(BTL_posterior)] = -np.inf
    Delta_posterior[np.isnan(Delta_posterior)] = -np.inf
    
    BTL_evidence = integrated_Likelihood(Y,BTL_posterior)
    Delta_evidence = integrated_Likelihood(Y,Delta_posterior)
    
    print("K = "+ str(np.exp(Delta_evidence-BTL_evidence)))

