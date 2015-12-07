#ifndef TYPES_H_
#define TYPES_H_

#include <stdint.h>

typedef uint_fast64_t chunk;
#define BITSPERCHUNK 64
#define ZERO 0ULL
#define ONE 1ULL

typedef unsigned dim;
typedef unsigned id;
#define MAXVAR 1100

typedef double value;

typedef struct {
	dim n, m, s, *d;
	id *vars;
	value *v;
} func;

#endif /* TYPES_H_ */
