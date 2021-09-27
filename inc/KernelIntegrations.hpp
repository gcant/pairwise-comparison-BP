#include <iostream>
#include <vector>
#include "cheby2.hpp"

cheby::ArrayXd Kernel(cheby::ArrayXd, floatT, floatT);

namespace KernelIntegration {
  
  typedef cheby::ArrayXd ArrayXd;
  
  ArrayXd coef_to_vals(ArrayXd const &c) {
    ArrayXd a = c;
    a[0] = 2.0*c[0];
    ArrayXd f = cheby::dct3(a);
    return f/2.0;
  }
  
  ArrayXd vals_to_coefs(ArrayXd const &f) {
    ArrayXd c = cheby::dct2(f) / f.size();
    c[0] /= 2.0;
    return c;
  }
  
  floatT integrate_kernel( ArrayXd const &f, ArrayXd const &K ) {
    ArrayXd c = vals_to_coefs(f * K);
    ArrayXd c_new = cheby::Chebyshev_coef_integrate(c);
    return cheby::Chebyshev_value(1.0,c_new);
  }
  
  ArrayXd K(ArrayXd x, floatT y, floatT beta) {
    return Kernel(x, y, beta);
  }
  
  ArrayXd T(ArrayXd x, floatT n) {
    return cos(n * acos(x));
  }
  
  std::vector<ArrayXd> compute_integrals(int L, floatT beta) {
  
    ArrayXd x = cheby::ChebPts(L);
  
    std::vector<ArrayXd> I_T(L);
    std::vector<ArrayXd> T_j(L);
    for (int i=0; i<L; ++i) {
      I_T[i].resize(L,0.0);
      T_j[i].resize(L,0.0);
    }
    for (int j=0; j<L; ++j) T_j[j] = T(x,j);
  
    for (int i=0; i<L; ++i) {
      ArrayXd K_i = K(x, x[i], beta);
      #pragma omp parallel shared(K_i, I_T, T_j) 
      for (int j=0; j<L; ++j) {
        I_T[j][i] = integrate_kernel(T_j[j], K_i);
      }
    }
  
    return I_T;
  }

};

