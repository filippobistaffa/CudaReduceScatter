#include "reducescatter.h"

void print(const func *f) {

	const char digits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

	for (dim i = 0; i < f->n; i++) {
		dim j = i, k = 0;
		do {
			printf("%c", digits[j % f->d[f->vars[k]]]);
			j /= f->d[f->vars[k]];
			k++;
		}
		while (j);
		if (f->m - k) printf("%0*d", (dim)(f->m - k), 0);
		printf(" = %f\n", f->v[i]);
	}
}

dim domainproduct(const id *vars, dim m, const dim *d) {

	register dim p = 1;
	for (dim i = 0; i < m; i++) p *= d[vars[i]];
	return p;
}
