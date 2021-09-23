// message update functions, along with one- and two-point entropies
#pragma once
#include "cheby2.hpp"
#include "DiGraph.h"
#include <unordered_map>
#include <vector>
#include <tuple>

typedef std::vector<std::pair<int,int>> elist_t;
typedef std::unordered_map<int, cheby::Chebyshev> marginal_t;
typedef std::unordered_map<int, marginal_t> message_t;

const bool KEEP_ORDER = true;
const bool ENFORCE_POSITIVE = true;


// create an undirected edgelist from a DiGraph – lists edges in both directions
elist_t DiGraph_to_list(DiGraph &G) {
  elist_t edges;
  for (auto i : G.nodes()) {
    for (auto j : G.successors(i)) {
      edges.push_back({i,j});
      edges.push_back({j,i});
    }
  }
  return edges;
}


floatT iterate_messages_delta_parallel(DiGraph const &G, elist_t const &m_order,
		message_t &messages, int L) {
  floatT delta = 0.0;
  #pragma omp parallel shared(m_order, messages, G, L) reduction(+: delta)
  for (int q=0; q<m_order.size(); ++q) {
    int i=m_order[q].first;
    int j=m_order[q].second;
    cheby::ArrayXd new_values(1.0,L+1-KEEP_ORDER);
    for (int k : G.successors(j)) if (k!=i)
      new_values *= messages.at(k).at(j).integral_vals(-1.0,KEEP_ORDER);
    for (int k : G.predecessors(j)) if (k!=i) 
      new_values *= -messages.at(k).at(j).integral_vals(1.0,KEEP_ORDER);
    cheby::Chebyshev new_message(cheby::Chebyshev_coefs_from_values(new_values)[std::slice(0,L,1)]);
    if (ENFORCE_POSITIVE) new_message.positives(0.0);
    new_message.normalize();
    delta += abs(messages[j][i].vals() - new_message.vals()).sum();
    messages[j][i] = new_message;
  }
  return delta;
}


marginal_t compute_marginals(DiGraph const &G, message_t &messages, int L){
  marginal_t marg;
  #pragma omp parallel for shared(marg, messages, G, L) 
  for (int i=0; i<G.number_of_nodes(); ++i) {
    cheby::ArrayXd marginal_values(1.0,L+1-KEEP_ORDER);
    for (int j : G.successors(i)) 
      marginal_values *= messages.at(j).at(i).integral_vals(-1.0,KEEP_ORDER);
    for (int j : G.predecessors(i)) 
      marginal_values *= -messages.at(j).at(i).integral_vals(1.0,KEEP_ORDER);
    cheby::Chebyshev new_marg(cheby::Chebyshev_coefs_from_values(marginal_values)[std::slice(0,L,1)]);
    if (ENFORCE_POSITIVE) new_marg.positives(0.0);
    new_marg.normalize();
    #pragma omp critical
    marg[i] = new_marg;
  }
  return marg;
}

template <class vec_t>
vec_t log0(vec_t X) {
  vec_t ans = X;
  for (int i=0; i<ans.size(); ++i) {
    if (ans[i]>0) ans[i] = log(ans[i]);
    else ans[i] = 0;
  }
  return ans;
}

floatT marginal_entropy(DiGraph const &G, marginal_t &marg, int L){
  floatT S = 0.0;
  int N = G.number_of_nodes();
  #pragma omp parallel for reduction(+: S)
  for (int i=0; i<N; ++i) {
    auto mu1 = marg.at(i);
    mu1.normalize();
    cheby::Chebyshev L(mu1.vals() * log0(mu1.vals()), INIT_BY_VALUES);
    floatT di = 0;
    di += G.in_degree(i);
    di += G.out_degree(i);
    S+= (di-1)*cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(L.coefs()));
  }
  return S;
}

floatT message_entropy(DiGraph const &G, message_t &messages, int L){
  floatT S = 0.0;
  int N = G.number_of_nodes();
  #pragma omp parallel for reduction(+: S)
  for (int i=0; i<N; ++i) {
    for (int j : G.successors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto C1  = -mu1.integrate(1.0, KEEP_ORDER);
      auto M2  = mu2.integrate(-1.0, KEEP_ORDER);
      cheby::Chebyshev Px( mu1.vals()*M2.vals(), INIT_BY_VALUES);
      cheby::Chebyshev Py( mu2.vals()*C1.vals(), INIT_BY_VALUES);
      cheby::ArrayXd cx_integral = cheby::Chebyshev_coef_integrate(Px.coefs());
      cheby::ArrayXd cy_integral = cheby::Chebyshev_coef_integrate(Py.coefs());
      floatT Zx = cheby::Chebyshev_value(1.0,cx_integral);
      floatT Zy = cheby::Chebyshev_value(1.0,cy_integral);
      Px /= Zx;
      Py /= Zy;
      cheby::Chebyshev Lx(Px.vals() * log0(mu1.vals()), INIT_BY_VALUES);
      cheby::Chebyshev Ly(Py.vals() * log0(mu2.vals()), INIT_BY_VALUES);
      floatT Sx = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Lx.coefs()));
      floatT Sy = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Ly.coefs()));
      S += Sx + Sy - 0.5* ( log(Zx) + log(Zy) );
    }
  }
  return S;
}


