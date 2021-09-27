#include <iostream>
#include <random>
#include "cheby2.hpp"
#include <unordered_map>
#include <vector>
#include <tuple>

typedef std::pair<cheby::Chebyshev, double> message_X;

std::random_device rd;
std::mt19937 rng(rd());
bool KEEP_ORDER=true;
std::vector<message_X> messages;

template <class vecT>
vecT log0(vecT X) {
  vecT ans = X;
  for (int i=0; i<ans.size(); ++i) {
    if (ans[i]>0) ans[i] = log(ans[i]);
    else ans[i] = 0;
  }
  return ans;
}

std::vector<int> poisson_vec(int n, double mu) {
  std::vector<int> ans(n);
  std::poisson_distribution<int> pois(mu);
  for (int i=0; i<n; ++i) ans[i] = pois(rng);
  return ans;
}

template <class T>
double nanmean(std::vector<T> const &X) {
  T ans = 0;
  double N = 0;
  for (int i=0; i<X.size(); ++i) {
    if (not(std::isnan(X[i]))) {
      ans += X[i];
      N++;
    }
  }
  return ((double)ans)/N;
}


void iterate_messages(std::vector<message_X> &messages, int L,
		std::vector<int> const &degrees) {

  int N = messages.size();
  std::uniform_int_distribution<int> unif_int(0,N-1);
  std::uniform_real_distribution<double> unif(0.0,1.0);

  #pragma omp parallel for shared(messages, unif_int, unif)
  for (int i=0; i<N; ++i) {
    cheby::ArrayXd new_values(1.0,L);
    double x = unif(rng);
    int d_i = degrees[i];
    while (d_i>0) {
      message_X message_in = messages.at( unif_int(rng) );
      if (message_in.second < x) {
        new_values *= message_in.first.integral_vals(-1.0,KEEP_ORDER);
	d_i--;
      }
      else if (message_in.second > x) {
        new_values *= -message_in.first.integral_vals(1.0,KEEP_ORDER);
	d_i--;
      }
    }
    cheby::Chebyshev new_message(cheby::Chebyshev_coefs_from_values(new_values));
    new_message.positives(10e-10);
    new_message.normalize();
    message_X m = {new_message, x};
    #pragma omp write
    messages[i] = m;
  }
}

floatT one_point_entropy(cheby::Chebyshev const &marg) {
  auto mu1 = marg;
  mu1.normalize();
  cheby::Chebyshev LN(mu1.vals() * log0(mu1.vals()), INIT_BY_VALUES);
  floatT S = cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(LN.coefs()));
  return S;
}

floatT two_point_entropy(message_X const &m1, message_X const &m2 ){
  cheby::Chebyshev mu1, mu2;
  if (m1.second < m2.second) {
    mu1 = m2.first;
    mu2 = m1.first;
  }
  else {
    mu1 = m1.first;
    mu2 = m2.first;
  }
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
  floatT S = Sx + Sy - 0.5* ( log(Zx) + log(Zy) );
  return S;
}

std::vector<message_X> random_marginals( std::vector<message_X> const &messages,
		std::vector<int> const &degrees, int L, int num_samples) {

  int N = messages.size();
  std::uniform_int_distribution<int> unif_int(0,N-1);
  std::uniform_real_distribution<double> unif(0.0,1.0);
  std::vector<message_X> ans(num_samples);

  #pragma omp parallel for shared(ans, unif_int, unif)
  for (int s=0; s<num_samples; ++s) {
    cheby::ArrayXd new_values(1.0,L);
    double x = unif(rng);
    int d_i = degrees[s];
    while (d_i>0) {
      message_X message_in = messages.at(unif_int(rng));
      if (message_in.second < x) {
        new_values *= message_in.first.integral_vals(-1.0,KEEP_ORDER);
	d_i--;
      }
      else if (message_in.second > x) {
        new_values *= -message_in.first.integral_vals(1.0,KEEP_ORDER);
	d_i--;
      }
    }
    cheby::Chebyshev new_message(cheby::Chebyshev_coefs_from_values(new_values));
    new_message.positives(0.0);
    new_message.normalize();
    #pragma omp write
    ans[s] = {new_message, x};
  }

  return ans;
}


floatT estimate_S(std::vector<message_X> const &messages, 
		std::vector<int> const &degrees, int L, double c) {

  std::uniform_int_distribution<int> unif_int(0,messages.size()-1);
  int num_samps = degrees.size();

  std::vector<message_X> one_point = random_marginals(messages, degrees, L, num_samps);
  floatT S1=0;
  #pragma omp parallel for reduction(+: S1)
  for (int i=0; i<one_point.size(); ++i) {
    S1 += (degrees[i]-1)*one_point_entropy(one_point[i].first)/num_samps;
  }

  floatT S2=0;
  #pragma omp parallel for reduction(+: S2)
  for (int i=0; i<num_samps; ++i) {
    auto m1 = messages.at(unif_int(rng));
    auto m2 = messages.at(unif_int(rng));
    S2 += two_point_entropy(m1,m2)/num_samps;
  }

  return (c/2.)*S2-S1;

}


extern "C" {

  void initPopulation(int pop_size, int L) {
    std::vector<message_X> new_pop(pop_size);
    for (int i=0; i<pop_size; ++i) new_pop[i] = {cheby::Chebyshev(L), i/pop_size};
    messages = new_pop;
  }
  
  void PoissonPopDyn(double mean_degree, int iterations, int entropy_samples, double *out) {
    int pop_size = messages.size();
    int L = messages[0].first.coefs().size();
    for (int s=0; s<iterations; ++s) {
      iterate_messages(messages, L, poisson_vec(pop_size,mean_degree));
      out[s] = (double) estimate_S(messages, poisson_vec(entropy_samples,mean_degree), L, mean_degree);
    }
  }

}

