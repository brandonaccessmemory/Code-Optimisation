#ifndef SPARSEMV_H
#define SPARSEMV_H
#include "mesh.h"

int sparsemv(struct mesh *restrict A, const float * const x, float * const y);
#endif
