#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include "sparsemv.h"
#include "omp.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *restrict A, const float * const x, float * const y)
{ 
  // initialising variables
  int i;
  register int loopFactor = 24;
  const int nrow = (const int) A->local_nrow;

  // multithreading the outerloop, splitting the initial chunk size of 200 to each individual thread
  #pragma omp parallel for schedule(guided)
  for(i=0;i<nrow;i++){
    float sum =0.0;
    // creating a vector to store total sum
    register __m256 sumVec = _mm256_set1_ps(0.0f);

    const float * const cur_vals = (const float * const) A->ptr_to_vals_in_row[i];
    const int * const cur_inds = (const int* const) A->ptr_to_inds_in_row[i];
    const int cur_nnz = (const int) A->nnz_in_row[i];

    // unrolling inner loop with loop factor 24
    int unroll = (cur_nnz/loopFactor)*loopFactor;
    for ( int j=0; j< unroll; j+= loopFactor) {
      // performing calculation on first 8 floats of the vector
      sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(_mm256_loadu_ps(cur_vals+j), _mm256_i32gather_ps(x,_mm256_loadu_si256((__m256i*)(cur_inds+j)),4)));
      // 8th - 16th float of the vector
      sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(_mm256_loadu_ps(cur_vals+j+8), _mm256_i32gather_ps(x,_mm256_loadu_si256((__m256i*)(cur_inds+j+8)),4)));
      // 16th - 24th float of the vector
      sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(_mm256_loadu_ps(cur_vals+j+16), _mm256_i32gather_ps(x,_mm256_loadu_si256((__m256i*)(cur_inds+j+16)),4)));
    }

    //adding each calculated sum of the vector to the variable sum
    sum += sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3] + sumVec[4] + sumVec[5] + sumVec[6] + sumVec[7];

    //clean up loop
    for (int j=unroll;j<cur_nnz;j++){
      sum += cur_vals[j]*x[cur_inds[j]];
    }
    y[i] = sum;
  }
  return 0;
}

