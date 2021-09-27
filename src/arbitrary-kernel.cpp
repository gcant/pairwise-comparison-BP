#include <random>
#include <iostream>
#include "KernelIntegrations.hpp"
#include "arbitrary-kernel-fns.hpp"

std::random_device rd;
std::mt19937 rng(rd());
floatT beta;

floatT ln_fact(floatT x) {
  if (x<=1) return 0;
  else return ln_fact(x-1) + log(x);
}

// This the the BTL kernel -- replace as needed 
floatT Kernel(floatT x, floatT y, floatT beta) {
	return 1.0 / (1.0 + exp(-beta*(x-y)));
}

cheby::ArrayXd Kernel(cheby::ArrayXd x, floatT y, floatT beta) {
	cheby::ArrayXd ans(x.size());
	for (int i=0; i<x.size(); ++i) ans[i] = Kernel(x[i], y, beta);
	return ans;
}

extern "C" {

    void compute_Cheb_pts(double *out, int L) {
      auto pts = cheby::ChebPts(L);
      for (int i=0; i<L; ++i) out[i] = (double)pts[i];
    }


  int full_algorithm(int num_nodes, int num_edges, int *edges, int L,
  		double beta_in, double tol, double *output, double *edgescores, int load_message) {

    beta = (floatT)beta_in;

    auto K = Kern_eval(L, beta);
    Eigen::MatrixXd I(L,L);
  
    {
      auto I_T = KernelIntegration::compute_integrals(L, beta);
      for (int i=0; i<L; ++i)
        for (int j=0; j<L; ++j)
          I(i,j) = I_T[j][i];
    }

    // construct graph
    DiGraph G;
    for (int i=0; i<num_nodes; ++i) G.add_node(i);
    for (int i=0; i<num_edges; ++i) G.add_edge(edges[2*i+1],edges[2*i]);
  
    elist_t m_order = DiGraph_to_list(G);  // message update order
    message_t messages;
    for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);

    int s=0;
    for (s; s<100; ++s) {
      std::shuffle(m_order.begin(), m_order.end(), rng);  // randomize update order 
      if (iterate_messages_delta_parallel_kernel(G, m_order, messages, L, I) < tol) {
        break;
      }
    }

    // compute one-node marginals
    marginal_t one_node_marginals = compute_marginals_kernel(G, messages, L, I);
  
    floatT U = energy_kernel(G, messages, L, K);   // two-point entropy
    floatT S2 = message_entropy_kernel(G, messages, L, K);   // two-point entropy
    floatT S1 = marginal_entropy(G, one_node_marginals, L); // one-point entropy
    floatT S = S2-S1;
    floatT lnZ = U - S;

    int t = 0;
    auto pts = cheby::ChebPts(L);
    for (t; t<pts.size(); ++t)  output[t] = (double) pts[t];
  
    for (int i=0; i<G.number_of_nodes(); ++i) {
      auto X = one_node_marginals[i].vals();
      for (int x=0; x<X.size(); ++x) {
        output[t] = (double) X[x];
        t++;
      }
    }

    output[t] = (double)S; ++t;
    output[t] = (double)lnZ;

    return s+1;
    }


}
