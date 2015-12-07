#include "reducescatter.h"

#include <cuda_runtime.h>
#include <vector_types.h>

template <typename type>
__attribute__((always_inline))
inline type power(dim base, unsigned exp) {

	register type result = 1;
	while (exp) {
		if (exp & 1) result *= base;
		exp >>= 1;
		base *= base;
	}
	return result;
}

__attribute__((always_inline))
inline dim swapcolumns(dim v, int3 *p, uint2 *d, dim n) {

	register dim nt = n, vt = v;
	register int cx, cy, a;
	register uint2 *dn = d;
	register int3 *pn = p;

	do {
		cx = (vt / pn->x) % dn->x;
		cy = (vt / pn->y) % dn->y;
		a = vt % pn->x;
		vt = a + cy * pn->x + (vt % pn->y - a - cx * pn->x + cx * pn->y) * dn->y / dn->x + vt - vt % pn->z;
		pn++;
		dn++;
	} while (--nt);

	return vt;
}

#include <omp.h>

#define SWAP(X, Y) do { register uint_fast64_t __t = (X); (X) = (Y); (Y) = __t; } while (0)

void shared2most(func *f, chunk m) {

	register dim x, y, n, i = 0;
	chunk s = ((ONE << f->s) - 1) << (f->m - f->s);
	if (s == m) return;
	chunk a = s & ~m;
	chunk o = m & ~s;
	int3 p[n = __builtin_popcountll(o)];
	uint2 d[n];

	#ifdef CUDAMALLOCHOST
	value *vt;
	cudaMallocHost((void **)&vt, sizeof(value) * f->n);
	#else
	value *vt = (value *)malloc(sizeof(value) * f->n);
	#endif

	do {
		x = __builtin_ctzll(o);
		y = __builtin_ctzll(a);
		p[i].x = domainproduct(f->vars, x, f->d);
		p[i].y = domainproduct(f->vars, y, f->d);
		p[i].z = domainproduct(f->vars, y + 1, f->d);
		d[i].x = f->d[f->vars[x]];
		d[i].y = f->d[f->vars[y]];
		SWAP(f->vars[x], f->vars[y]);
		o ^= ONE << x;
		a ^= ONE << y;
		i++;
	} while (o);

	#pragma omp parallel for private(i)
	for (i = 0; i < f->n; i++) vt[swapcolumns(i, p, d, n)] = f->v[i];

	#ifdef CUDAMALLOCHOST
	cudaFreeHost(f->v);
	#else
	free(f->v);
	#endif

	f->v = vt;
}

void reordershared(func *f1, func *f2) {

	#ifdef CUDAMALLOCHOST
	value *vt;
	cudaMallocHost((void **)&vt, sizeof(value) * f2->n);
	#else
	value *vt = (value *)malloc(sizeof(value) * f2->n);
	#endif

	int3 p[f2->s];
	uint2 d[f2->s];
	register dim i, j, k;
	dim *v = (dim *)malloc(sizeof(dim) * MAXVAR);

	for (i = 0; i < f2->m; i++) v[f2->vars[i]] = i;
	//v[k] = index of variable "k" in f2->vars
	for (i = 0; i < f2->s; i++) {
		k = f2->m - 1 - i;
		j = v[f1->vars[f1->m - 1 - i]];
		p[i].x = domainproduct(f2->vars, j, f2->d);
		p[i].y = domainproduct(f2->vars, k, f2->d);
		p[i].z = domainproduct(f2->vars, k + 1, f2->d);
		d[i].x = f2->d[f2->vars[j]];
		d[i].y = f2->d[f2->vars[k]];
		SWAP(f2->vars[k], f2->vars[j]);
		SWAP(v[f2->vars[k]], v[f2->vars[j]]);
	}

	#pragma omp parallel for private(i)
	for (i = 0; i < f2->n; i++) vt[swapcolumns(i, p, d, f2->s)] = f2->v[i];

	#ifdef CUDAMALLOCHOST
	cudaFreeHost(f2->v);
	#else
	free(f2->v);
	#endif

	f2->v = vt;
}

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2) {

        f1->s = f2->s = 0;
	*s1 = *s2 = ZERO;

        for (dim i = 0; i < f1->m; i++)
                for (dim j = 0; j < f2->m; j++)
                        if (f1->vars[i] == f2->vars[j]) {
                                *s1 |= ONE << i;
                                *s2 |= ONE << j;
                                (f1->s)++;
                                (f2->s)++;
                                break;
                        }
}
