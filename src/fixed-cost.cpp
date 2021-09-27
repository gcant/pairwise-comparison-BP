#include <random>
#include <iostream>
#include "fixed-cost-fns.hpp"

std::random_device rd;
std::mt19937 rng(rd());

floatT Delta;
cheby::ArrayXd prior;
message_t saved_messages;

bool USE_PRIOR=false;

extern "C" {

  void set_prior(double *new_prior, int L) {
    USE_PRIOR=true;
    cheby::ArrayXd temp(0.0,L);
    for (int i=0; i<L; ++i) temp[i] = (floatT)new_prior[i];
    prior = temp;
  }

  void compute_Cheb_pts(double *out, int L) {
    auto pts = cheby::ChebPts(L);
    for (int i=0; i<L; ++i) out[i] = (double)pts[i];
  }


  int full_algorithm(int num_nodes, int num_edges, int *edges, int L,
  		double delta_in, double tol, double *output, double *edgescores, int load_message) {

    Delta = (floatT)delta_in;

    // note: should adjust how priors are handled
    // prior = create_prior(L);
  
    // construct graph
    DiGraph G;
    for (int i=0; i<num_nodes; ++i) G.add_node(i);
    for (int i=0; i<num_edges; ++i) G.add_edge(edges[2*i+1],edges[2*i]);
  
    elist_t m_order = DiGraph_to_list(G);  // message update order
    message_t messages;
    for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);
    if (load_message) messages = saved_messages;
  
    int s=0;
    for (s; s<100; ++s) {
      std::shuffle(m_order.begin(), m_order.end(), rng);  // randomize update order 
      if (iterate_messages_delta_parallel(G, m_order, messages, L) < tol) {
        break;
      }
      if (std::isnan(messages[m_order[0].first][m_order[0].second].vals()[0])) {
        std::cerr << "Error...restarting." << std::endl;
        for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);
      }
    }
  
    // compute one-node marginals
    marginal_t one_node_marginals = compute_marginals(G, messages, L);
  
    double S2 = message_entropy(G, messages, L);   // two-point entropy
    double S1 = marginal_entropy(G, one_node_marginals, L); // one-point entropy
    double logV = -(S2-S1) - G.number_of_nodes()*log(2);   // volume

    double EV = expected_violations(G, messages, L);
    double UP = 0.;
    if (USE_PRIOR) UP = prior_energy(G, one_node_marginals, L);
  
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

    output[t] = UP; ++t;
  
    output[t] = S1; t++;
    output[t] = S2; t++;
  
    output[t] = -(S2-S1); ++t;
    output[t] = logV; ++t;
  
    output[t] = EV; ++t;


    for (int e=0; e<num_edges; ++e) {
      int i = edges[2*e+1];
      int j = edges[2*e];
      edgescores[e] = (double) violation_score(i,j,messages,L);
    }

    saved_messages = messages;

    return s+1;
  
  }

}
