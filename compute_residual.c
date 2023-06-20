#include <math.h>  // needed for fabs
#include "compute_residual.h"

/**
 * @brief Compute the 1-norm difference between two vectors
 * 
 * @param n Number of vector elements
 * @param v1 Input vector
 * @param v2 Input vector
 * @param residual Pointer to scalar return value
 * @return int 0 if no error
 */
int compute_residual(const int n, const float * const v1, const float * const v2, float * const residual)
{
  float local_residual = 0.0;
  
  for (int i=0; i<n; i++) {
    float diff = fabs(v1[i] - v2[i]);
    if (diff > local_residual) {
      local_residual = diff;
    }
  }

  *residual = local_residual;

  return 0;
}
