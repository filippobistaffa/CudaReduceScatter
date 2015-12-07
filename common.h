#ifndef COMMON_H_
#define COMMON_H_

void print(const func *f);

dim domainproduct(const id *vars, dim m, const dim *d);

#include <iostream>
template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name) {

	printf("%s = [ ", name);
	while (n--) std::cout << *(buf++) << " ";
	printf("]\n");
}

#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))
#define BREAKPOINT(MSG) do { puts(MSG); fflush(stdout); while (getchar() != '\n'); } while (0)

#endif  /* COMMON_H_ */
