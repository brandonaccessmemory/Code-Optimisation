#include "waxpby.h"
#include <immintrin.h>
#include "omp.h"
/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const float alpha, const float * const x, const float beta, const float * const y, float * const w) {  
  // initialising variables and registers
  int i;
  int loopFactor = 8;
  int loopN = (n/loopFactor)*loopFactor;
  // creating a vector with all values set to a constant 
  register __m256 alphaVec = _mm256_set1_ps(alpha);
  register __m256 betaVec = _mm256_set1_ps(beta);

  // alpha is set to 1, hence making the if statements redundant, however the arguments to waxpby might be different during testing, so it was left in instead
  // multi-thread unrolled loop
  #pragma omp parallel for schedule(static) firstprivate(loopN,loopFactor)
  for (i = 0; i < loopN ; i+= loopFactor){
    if (alpha == 1.0){
      // storing the results into the output vector
      _mm256_store_ps(w+i,_mm256_add_ps(_mm256_load_ps(x+i), _mm256_mul_ps(betaVec,_mm256_load_ps(y+i))));
    } else if (beta == 1.0){      
      _mm256_store_ps(w+i,_mm256_add_ps(_mm256_load_ps(y+i), _mm256_mul_ps(alphaVec,_mm256_load_ps(x+i))));
    } else {
      _mm256_store_ps(w+i,_mm256_add_ps(_mm256_mul_ps(alphaVec, _mm256_load_ps(x+i)), _mm256_mul_ps(betaVec,_mm256_load_ps(y+i))));
    }
  }

  // clean up loop
  #pragma omp parallel for 
  for (i=loopN;i<n;i++){
    if (alpha == 1.0){
      w[i] = x[i] + beta * y[i];
    } else if (beta == 1.0){
      w[i] = alpha * x[i] + y[i];
    } else {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
  
  return 0;
}
