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
//const bool USE_PRIOR = true;
extern bool USE_PRIOR;
extern floatT Delta;
extern cheby::ArrayXd prior;

cheby::ArrayXd create_prior(int N) {
	cheby::ArrayXd P(0.0, N);
	auto pts = cheby::ChebPts(N);
	for (int i=0; i<N; ++i) {
		P[i] = exp(-2.0*pts[i]*pts[i]);
	}
	return P;
}

// create an undirected edgelist from a DiGraph – lists edges in both directions(!!)
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
    if (USE_PRIOR) new_values = prior;
    for (int k : G.successors(j)) if (k!=i)
      new_values *= ((1.-Delta)*messages.at(k).at(j).integral_vals(-1.0,KEEP_ORDER)) + Delta;
    for (int k : G.predecessors(j)) if (k!=i) 
      new_values *= ((1.-Delta)*(-messages.at(k).at(j).integral_vals(1.0,KEEP_ORDER))) + Delta;
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
  for (int i : G.nodes()) {
    marg[i] = cheby::Chebyshev(L);
    cheby::ArrayXd marginal_values(1.0,L+1-KEEP_ORDER);
    if (USE_PRIOR) marginal_values = prior;
    for (int j : G.successors(i)) 
      marginal_values *= ((1.-Delta)*messages.at(j).at(i).integral_vals(-1.0,KEEP_ORDER)) + Delta;
    for (int j : G.predecessors(i)) 
      marginal_values *= ((1.-Delta)*(-messages.at(j).at(i).integral_vals(1.0,KEEP_ORDER))) + Delta;
    marg[i].set_coefs(cheby::Chebyshev_coefs_from_values(marginal_values)[std::slice(0,L,1)]);
    if (ENFORCE_POSITIVE) marg[i].positives(0.0);
    marg[i].normalize();
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

floatT xlogx(floatT X) {
  if (X>0) return X*log(X);
  else return 0.0;
}

floatT marginal_entropy(DiGraph const &G, marginal_t &marg, int L){
  floatT S = 0.0;
  for (int i : G.nodes()) {
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
  for (int i : G.nodes()) {
    for (int j : G.successors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto C1  = -mu1.integrate(1.0, KEEP_ORDER);
      auto M2  = mu2.integrate(-1.0, KEEP_ORDER);
      cheby::Chebyshev Px( mu1.vals()*((1.-Delta)*M2.vals() + Delta), INIT_BY_VALUES);
      cheby::Chebyshev Py( mu2.vals()*((1.-Delta)*C1.vals() + Delta), INIT_BY_VALUES);
      cheby::ArrayXd cx_integral = cheby::Chebyshev_coef_integrate(Px.coefs());
      cheby::ArrayXd cy_integral = cheby::Chebyshev_coef_integrate(Py.coefs());
      floatT Zx = cheby::Chebyshev_value(1.0,cx_integral);
      floatT Zy = cheby::Chebyshev_value(1.0,cy_integral);
      //std::cout << Zx << ", " << Zy << std::endl;
      Px /= Zx;
      Py /= Zy;

      auto M1  = mu1.integrate(-1.0, KEEP_ORDER);
      cheby::Chebyshev Pyd( mu2.vals()*M1.vals(), INIT_BY_VALUES);
      cheby::ArrayXd cyd_integral = cheby::Chebyshev_coef_integrate(Pyd.coefs());
      floatT Zyd = cheby::Chebyshev_value(1.0,cyd_integral);
      Zyd /= (0.5*Zx + 0.5*Zy);
      floatT Syd = xlogx(Delta) * Zyd;

      cheby::Chebyshev Lx(Px.vals() * log0(mu1.vals()), INIT_BY_VALUES);
      cheby::Chebyshev Ly(Py.vals() * log0(mu2.vals()), INIT_BY_VALUES);
      floatT Sx = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Lx.coefs()));
      floatT Sy = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Ly.coefs()));
      S += Sx + Sy + Syd - 0.5* ( log(Zx) + log(Zy) );
    }
  }
  return S;
}

floatT violation_score(int i, int j, message_t &messages, int L) {
  floatT V = 0.0;
  auto mu1 = messages.at(i).at(j);
  auto mu2 = messages.at(j).at(i);
  mu1.normalize();
  mu2.normalize();
  auto M1  = mu1.integrate(-1.0, KEEP_ORDER);
  auto M2  = mu2.integrate(-1.0, KEEP_ORDER);
  cheby::Chebyshev Q1( mu1.vals()*M2.vals(), INIT_BY_VALUES);
  cheby::Chebyshev Q2( M1.vals()*mu2.vals(), INIT_BY_VALUES);
  floatT Z1 = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Q1.coefs()));
  floatT Z2 = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Q2.coefs()));
  V += (Delta*Z2) / (Z1+Delta*Z2);
  return V;
}

floatT expected_violations(DiGraph const &G, message_t &messages, int L) {
  floatT V = 0.0;
  for (int i : G.nodes()) {
    for (int j : G.successors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto M1  = mu1.integrate(-1.0, KEEP_ORDER);
      auto M2  = mu2.integrate(-1.0, KEEP_ORDER);

      cheby::Chebyshev Q1( mu1.vals()*M2.vals(), INIT_BY_VALUES);
      cheby::Chebyshev Q2( M1.vals()*mu2.vals(), INIT_BY_VALUES);

      floatT Z1 = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Q1.coefs()));
      floatT Z2 = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(Q2.coefs()));

      V += (Delta*Z2) / (Z1+Delta*Z2);

    }
  }
  return V;
}


floatT prior_energy(DiGraph const &G, marginal_t &marg, int L){
  floatT U = 0.0;
  for (int i : G.nodes()) {
    auto mu1 = marg.at(i);
    mu1.normalize();
    cheby::Chebyshev L(mu1.vals() * log0(prior), INIT_BY_VALUES);
    U += cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(L.coefs()));
  }
  return U;
}

