#ifndef HPCCG_H
#define HPCCG_H
#include <string.h>

#include "sparsemv.h"
#include "ddot.h"
#include "waxpby.h"
#include "mesh.h"

int conjugateGradient(struct mesh * A,
	  const float * const b, float * const x,
	  const int max_iter, const float tolerance, int *niters, float *normr, double * times,
	  char* siloName);
#endif
