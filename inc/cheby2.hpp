#pragma once
#include <valarray>
#include "r2r.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

const int INIT_BY_COEFS = 0;
const int INIT_BY_VALUES = 1;

namespace cheby {

  typedef std::valarray<floatT> ArrayXd;
  
  // discrete cosine transform function
  r2r::r2r dct2(FFTW_REDFT10);
  r2r::r2r dct3(FFTW_REDFT01);
  
  // adapted from numpy
  // compute the value of the Chebyshev polynomial with coefficients c,
  // evaluated at point x
  template <class vec_t>
  floatT Chebyshev_value(floatT x, vec_t const &c) {
    floatT c0,c1;
    if (c.size()==1) {
      c0 = c[0];
      c1 = 0;
    }
    else if (c.size()==2) {
      c0 = c[0];
      c1 = c[1];
    }
    else {
      floatT x2 = x*2.0;
      c0 = c[c.size()-2];
      c1 = c[c.size()-1];
      for (int i=3; i < c.size()+1; ++i){
        floatT tmp = c0;
        c0 = c[c.size()-i] - c1;
        c1 = tmp + c1*x2;
      }
    }
    return c0 + c1*x;
  }
  
  // adapted from ChebTools
  // compute the coefficients for the integrated Chebyshev polynomial with
  // coefficients m_c
  // lbnd is the lower bound for the definite integral
  // if keep_order, only return coefficients to the same order as m_c
  ArrayXd Chebyshev_coef_integrate(ArrayXd const &m_c, floatT lbnd=-1.0,
        	  bool keep_order=false) {
  
    //Eigen::ArrayXd c = Eigen::ArrayXd::Zero(m_c.size()+1);
    ArrayXd c(0.0,m_c.size()+1);
  
    c[1] = (2.0*m_c[0] - m_c[2]) / 2.0;
    for (int i=2; i<m_c.size()-1; ++i) {
      c[i] = (m_c[i - 1] - m_c[i + 1]) / (2.0 * i);
    }
    for (int i=m_c.size()-1; i<m_c.size()+1; ++i) {
      c[i] = (m_c[i - 1]) / (2.0 * i);
    }
    c[0] = -Chebyshev_value(lbnd,c);
  
    if (keep_order) c = (ArrayXd)c[std::slice(0,m_c.size(),1)];
  
    return c;
  }
  
  // compute the Chebyshev coefficients from the values of the function at
  // the Chebyshev points, i.e. cos( k pi / N )
  ArrayXd Chebyshev_coefs_from_values(ArrayXd const &f) {
    // CHANGED
    ArrayXd c = dct2(f);
    c /= c.size();
    c[0] /= 2.0;
    return c;
  }

  ArrayXd ChebPts(int N) {
    // CHANGED
    ArrayXd pts(0.0,N);
    floatT pi_twoN = M_PI / (2.0*N);
    for (int k=0; k<N; ++k)
      pts[k] = cos( (2.0*k + 1.0) * pi_twoN );
    return pts;
  }
    
  
  class Chebyshev {
    private:
      ArrayXd c;  // coefficients
      ArrayXd f;  // function value at nodes
    public:
      Chebyshev(int);
      Chebyshev(ArrayXd const &, int);
      Chebyshev(ArrayXd const &, ArrayXd const &);
      floatT value(floatT x) {return Chebyshev_value(x, c);};
      void update_coefs(void);
      void update_vals(void);
      void normalize(floatT);
      void positives(floatT);
      void set_vals(ArrayXd const &);
      void set_coefs(ArrayXd const &);
      ArrayXd coefs(void){return c;};
      ArrayXd vals(void){return f;};
      ArrayXd integral_vals(floatT, bool);
      Chebyshev integrate(floatT, bool);
  
      Chebyshev operator-();
      void operator/=(floatT);
  };
    
  Chebyshev::Chebyshev(int n=4) {
    c.resize(n,0.0);
    c[0] = 1.0;
    f.resize(n,1.0);
  }
    
  Chebyshev::Chebyshev(ArrayXd const &a, int init_cond=0) {
    if (init_cond == INIT_BY_VALUES) {
      f = a;
      update_coefs();
    }
    else {
      c = a;
      update_vals();
    }
  }
  
  Chebyshev::Chebyshev(ArrayXd const &c_in, ArrayXd const &f_in) {
    c = c_in;
    f = f_in;
  }
  
  void Chebyshev::update_coefs(void){
    // CHANGED
    c = dct2(f);
    c /= c.size();
    c[0] /= 2.0;
  }
  
  void Chebyshev::update_vals(void){
    // CHANGED
    ArrayXd c_transform = c;
    c_transform[0] *= 2.0;
    f = dct3(c_transform) / 2.0;
  }
    
  // rescale so that integral over [-1,1] = 1
  void Chebyshev::normalize(floatT value=1.0) {
    ArrayXd c_integral = Chebyshev_coef_integrate(c);
    floatT Z = value/Chebyshev_value(1.0,c_integral);
    c *= Z;
    f *= Z;
  }
  
  void Chebyshev::positives(floatT value=10e-16) {
    ArrayXd f_new = f;
    for (int i=0; i<f_new.size(); ++i)
      if (f_new[i]<0.0) f_new[i]=value;
    f = f_new;
    //f = abs(f);
    update_coefs();
  }
  

  
  ArrayXd Chebyshev::integral_vals(floatT lbnd=-1.0, bool keep_order=false) {
    ArrayXd c_integral = Chebyshev_coef_integrate(c, lbnd, keep_order);
    c_integral[0] *= 2.0;
    return dct3(c_integral) / 2.0;
  }

  
  Chebyshev Chebyshev::integrate(floatT lbnd=-1.0, bool keep_order=false) {
    ArrayXd c_integral = Chebyshev_coef_integrate(c, lbnd, keep_order);
    Chebyshev ans(c_integral, INIT_BY_COEFS);
    return ans;
  }
  
  void Chebyshev::set_vals(ArrayXd const &new_vals) {
    f = new_vals;
    update_coefs();
  }
  
  void Chebyshev::set_coefs(ArrayXd const &new_coefs) {
    c = new_coefs;
    update_vals();
  }


  Chebyshev Chebyshev::operator-() {
    Chebyshev ans(-c,-f);
    return ans;
  }

  void Chebyshev::operator/=(floatT Z) {
    c /= Z;
    f /= Z;
  }
}

