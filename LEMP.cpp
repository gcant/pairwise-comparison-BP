// main BP algorithm, with C compatibility
#include <random>
#include <iostream>
#include "cheby-mp.hpp"

std::random_device rd;
std::mt19937 rng(rd());

extern "C" {

  int full_algorithm(int num_nodes, int num_edges, int *edges, int L,
  		double tol, double *output) {
  
    // construct graph from *edges
    DiGraph G;
    for (int i=0; i<num_nodes; ++i) G.add_node(i);
    for (int i=0; i<num_edges; ++i) G.add_edge(edges[2*i+1],edges[2*i]);
  
    // initialize the messages
    elist_t m_order = DiGraph_to_list(G);  // message update order
    message_t messages;
    for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);

    // main BP loop
    int s=0;
    for (s; s<201; ++s) {
      std::shuffle(m_order.begin(), m_order.end(), rng);  // randomize update order 
      if (iterate_messages_delta_parallel(G, m_order, messages, L) < tol)
        break;
      if (std::isnan(messages[m_order[0].first][m_order[0].second].vals()[0])) {
        std::cerr << "Error...restarting." << std::endl;
        for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);
      }
    }
  
    // compute one-node marginals
    marginal_t one_node_marginals = compute_marginals(G, messages, L);
  
    double S2 = message_entropy(G, messages, L);   // two-point entropy
    double S1 = marginal_entropy(G, one_node_marginals, L); // one-point entropy
    double logV = -(S2-S1) - G.number_of_nodes()*log(2);   // volume of polytope
  
    // save the marginals to output array
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
  
    // save various quantites of interest to output array
    output[t] = S1; t++;
    output[t] = S2; t++;
    output[t] = -(S2-S1); ++t;
    output[t] = logV; ++t;
  
    return s+1;
  }
}
