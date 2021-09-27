#pragma once
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include "LEMP-fns.hpp"

floatT Kernel(floatT, floatT, floatT);
cheby::ArrayXd Kernel(cheby::ArrayXd, floatT, floatT);

Eigen::MatrixXd load_matrix(std::string f_name, int L) {
  Eigen::MatrixXd X(L,L);
  std::ifstream in_file(f_name);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      in_file >> X(i,j);
    }
  }
  return X;
}

cheby::ArrayXd dot(Eigen::MatrixXd const &X, cheby::ArrayXd const &a) {
  int L = a.size();
  cheby::ArrayXd ans(0.0,L);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      ans[i] += X(i,j)*a[j];
    }
  }
  return ans;
}

floatT oneD_integral( cheby::ArrayXd const &f ) {
	cheby::Chebyshev f_cheb(f, INIT_BY_VALUES);
	return cheby::Chebyshev_value(1.0, cheby::Chebyshev_coef_integrate(f_cheb.coefs()));
}


floatT twoD_integral( std::vector<cheby::ArrayXd> const &f ) {
	int L = f.size();
	cheby::ArrayXd g(0.0,L);
	for (int k=0; k<L; ++k) {
		g[k] = oneD_integral(f[k]);
	}
	return oneD_integral(g);
}


std::vector<cheby::ArrayXd> Kern_eval(int L, floatT beta) {
	std::vector<cheby::ArrayXd> ans(L);
	for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
	cheby::ArrayXd X = cheby::ChebPts(L);
	for (int i=0; i<L; ++i) {
		for (int j=0; j<L; ++j) {
			ans[i][j] = Kernel(X[i],X[j],beta);
		}
	}
	return ans;
}

floatT xlogx(floatT x){
	if (x>0) return x * log(x);
	else return 0.0;
}

std::vector<cheby::ArrayXd> KlnK_eval(int L, floatT beta) {
	std::vector<cheby::ArrayXd> ans(L);
	for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
	cheby::ArrayXd X = cheby::ChebPts(L);
	for (int i=0; i<L; ++i) {
		for (int j=0; j<L; ++j) {
			ans[i][j] = xlogx(Kernel(X[i],X[j],beta));
		}
	}
	return ans;
}

std::vector<cheby::ArrayXd> two_point_dist( cheby::ArrayXd const &f1, 
		cheby::ArrayXd const &f2, std::vector<cheby::ArrayXd> const &K ) {

	int L = f1.size();
	std::vector<cheby::ArrayXd> ans(L);
	for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
	for (int i=0; i<L; ++i) {
		for (int j=0; j<L; ++j) {
			ans[i][j] = K[i][j]*f1[i]*f2[j];
		}
	}
	return ans;

}

floatT message_entropy_kernel(DiGraph const &G, message_t &messages, int L,
		 std::vector<cheby::ArrayXd> const &K) {
  floatT S = 0.0;
  for (int i : G.nodes()) {
    for (int j : G.successors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();

      std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);
      auto QlnQ = Q;
      for (int ii=0; ii<L; ++ii) QlnQ[ii] = Q[ii] * log0(Q[ii]);

      floatT Z = twoD_integral(Q);
      S -= log(Z) - (twoD_integral(QlnQ)/Z);
      //S += - (twoD_integral(QlnQ)/Z);
    }
  }
  return S;
}

floatT energy_kernel(DiGraph const &G, message_t &messages, int L,
		 std::vector<cheby::ArrayXd> const &K) {
  floatT U = 0.0;
  for (int i : G.nodes()) {
    for (int j : G.successors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();

      std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);
      auto QlnK = Q;
      for (int ii=0; ii<L; ++ii) QlnK[ii] = Q[ii] * log0(K[ii]);

      floatT Z = twoD_integral(Q);
      U += twoD_integral(QlnK)/Z;
    }
  }
  return U;
}


floatT iterate_messages_delta_parallel_kernel(DiGraph const &G, elist_t const &m_order,
		message_t &messages, int L, Eigen::MatrixXd const &I) {
  floatT delta = 0.0;
  #pragma omp parallel shared(m_order, messages, G, L, I) reduction(+: delta)
  for (int q=0; q<m_order.size(); ++q) {
    int i=m_order[q].first;
    int j=m_order[q].second;
    cheby::ArrayXd new_values(1.0,L);
    for (int k : G.successors(j)) if (k!=i)
      new_values *= dot(I,messages.at(k).at(j).coefs());
    for (int k : G.predecessors(j)) if (k!=i) 
      new_values *= -dot(I,messages.at(k).at(j).coefs()) + 1.0;
    cheby::Chebyshev new_message(new_values,INIT_BY_VALUES);
    if (ENFORCE_POSITIVE) new_message.positives(0.0);
    new_message.normalize();
    //if (j==0) new_message = zero_marg(L);
    delta += abs(messages[j][i].vals() - new_message.vals()).sum();
    messages[j][i] = new_message;
  }
  return delta;
}


marginal_t compute_marginals_kernel(DiGraph const &G, message_t &messages, int L, 
		Eigen::MatrixXd const &I) {
  marginal_t marg;
  for (int i : G.nodes()) {
    marg[i] = cheby::Chebyshev(L);
    cheby::ArrayXd marginal_values(1.0,L);
    for (int j : G.successors(i)) 
      marginal_values *= dot(I,messages.at(j).at(i).coefs());
    for (int j : G.predecessors(i)) 
      marginal_values *= -dot(I,messages.at(j).at(i).coefs()) + 1.0;
    marg[i].set_coefs(cheby::Chebyshev_coefs_from_values(marginal_values)[std::slice(0,L,1)]);
    if (ENFORCE_POSITIVE) marg[i].positives(0.0);
    marg[i].normalize();
  }
  return marg;
}

