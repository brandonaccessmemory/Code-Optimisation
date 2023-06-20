#include "ddot.h"
#include <immintrin.h>
#include "omp.h"
#include <stdio.h>
/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const float * const x, const float * const y, float * const result) {  
  //initialise variables and registers
  int i;
  float local_result = 0.0;
  int loopFactor = 8;
  int loopN = (n/loopFactor)*loopFactor;
  // creating a vector with all values set to a constant 
  __m256 v_shared = _mm256_set1_ps(0.0f);
  __m256 v_individual = _mm256_set1_ps(0.0f);

  // multi-thread unrolled loop, v_shared is the shared total sum, v_individual is the individual sum of each thread
  #pragma omp parallel shared(v_shared) firstprivate(v_individual)
  {
    #pragma omp for
    for ( i = 0; i < loopN ; i += loopFactor){
      // perform required calculations
      v_individual = _mm256_add_ps(v_individual,_mm256_mul_ps(_mm256_load_ps(x+i),_mm256_load_ps(y+i)));
    }
    // avoiding race condition
    #pragma omp critical 
    {
      // adding up the individual sum calculated by each thread to the shared variable
      v_shared = _mm256_add_ps(v_shared,v_individual);
    }
  }

  // adding the total sum to the variable local_result
  local_result += v_shared[0] + v_shared[1] + v_shared[2] + v_shared[3] + v_shared[4] + v_shared[5] + v_shared[6] + v_shared[7];
  
  // clean up loop, using reduction to avoid race condition
  #pragma omp parallel for reduction(+:local_result)
  for( i =loopN; i<n;i++) {
    local_result += x[i]*y[i];
  }

  *result = local_result;
  return 0;
}


