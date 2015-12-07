#include "reducescatter.h"

using namespace cub;

__global__ void reducescatter(value *v1d, value *v2d, value *sep, uint n1, uint n2) {

	/*
	assert(n1 >= THREADS * ITEMS);
	assert(n1 % (THREADS * ITEMS) == 0);
	assert(n2 >= THREADS * ITEMS);
	assert(n2 % (THREADS * ITEMS) == 0);
	*/

	register uint tx = threadIdx.x, bx = blockIdx.x;
	register uint i, m1 = n1 / THREADS, m2 = n2 / THREADS;
	typedef BlockReduce<value, THREADS, ALGORITHM> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage shared;
	__shared__ value sepratio;
	register value reduction = 0; // Identity element for reduction
	register value data[1];

	for (i = 0; i < m1; i++) {
		LOADDIRECTBLOCKED(tx, v1d + bx * n1 + i * THREADS, data);
		reduction += BlockReduceT(shared).Sum(data);
		__syncthreads();
	}

	if ((m1 = n1 % THREADS)) {
		LoadDirectBlocked(tx, v1d + bx * n1 + i * THREADS, data, m1);
		reduction += BlockReduceT(shared).Sum(data[0], m1);
		__syncthreads();
	}

	if (!tx) {
		sepratio = max((value)0, reduction / sep[bx]); // Old separator value
		sep[bx] = reduction; // Update separator
	}
	__syncthreads();

	for (i = 0; i < m2; i++) {
		LOADDIRECTBLOCKED(tx, v2d + bx * n2 + i * THREADS, data);
		data[0] *= sepratio;
		STOREDIRECTBLOCKED(tx, v2d + bx * n2 + i * THREADS, data);
	}

	if ((m2 = n2 % THREADS)) {
		LoadDirectBlocked(tx, v2d + bx * n2 + i * THREADS, data, m2);
		if (tx < m2) data[0] *= sepratio;
		StoreDirectBlocked(tx, v2d + bx * n2 + i * THREADS, data, m2);
	}
}

void updatepotential(func *f1, func *f2, func *sep, const dim *domains, value *f2sum, value *sepsum, size_t *transfer, size_t *preprocess) {

	//printbuf(f1->vars, f1->m, "f1->vars");
	//print(f1);
	//printbuf(f2->vars, f2->m, "f2->vars");
	//print(f2);
	//printbuf(sep->vars, sep->m, "sep->vars");
	//print(sep);

	struct timeval t1, t2;
	chunk c1, c2;
	gettimeofday(&t1, NULL);
	sharedmasks(f1, &c1, f2, &c2);
	shared2most(f1, c1);
	reordershared(f1, f2);
	gettimeofday(&t2, NULL);
	(*preprocess) += (t2.tv_usec - t1.tv_usec) / 1e3 + (t2.tv_sec - t1.tv_sec) * 1e3;

	//printbuf(f1->vars, f1->m, "f1->vars");
	//print(f1);
	//printbuf(f2->vars, f2->m, "f2->vars");
	//print(f2);
	//printbuf(sep->vars, sep->m, "sep->vars");
	//print(sep);

	value *v1d, *v2d, *sd;
	cudaMalloc(&v1d, sizeof(value) * f1->n);
        cudaMalloc(&v2d, sizeof(value) * f2->n);
        cudaMalloc(&sd, sizeof(value) * sep->n);

	dim n1, n2, ns = CEIL(sep->n, MAXBLOCKS);
	n1 = domainproduct(f1->vars, f1->m - f1->s, domains);
	n2 = domainproduct(f2->vars, f2->m - f1->s, domains);
	//printf("Each of the %u blocks has to reduce %u rows and product %u rows\n", sep->n, n1, n2);
	cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ns);
	dim *blocks = (dim *)malloc(sizeof(dim) * ns);

	for (dim i = 0; i < ns - 1; i++) blocks[i] = MAXBLOCKS;
	blocks[ns - 1] = (sep->n % MAXBLOCKS) ? (sep->n % MAXBLOCKS) : MAXBLOCKS;
	//printbuf(blocks, ns, "Blocks");
	gettimeofday(&t1, NULL);

	for (dim i = 0; i < ns; i++) {
		cudaStreamCreate(streams + i);
		cudaMemcpyAsync(v1d + i * MAXBLOCKS * n1, f1->v + i * MAXBLOCKS * n1, sizeof(value) * n1 * blocks[i], cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(v2d + i * MAXBLOCKS * n2, f2->v + i * MAXBLOCKS * n2, sizeof(value) * n2 * blocks[i], cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(sd + i * MAXBLOCKS, sep->v + i * MAXBLOCKS, sizeof(value) * blocks[i], cudaMemcpyHostToDevice, streams[i]);
	}

	gettimeofday(&t2, NULL);
	(*transfer) += (t2.tv_usec - t1.tv_usec) / 1e3 + (t2.tv_sec - t1.tv_sec) * 1e3;

	for (dim i = 0; i < ns; i++)
		reducescatter<<<blocks[i], THREADS, 0, streams[i]>>>(v1d + MAXBLOCKS * i * n1, v2d + MAXBLOCKS * i * n2, sd + MAXBLOCKS * i, n1, n2);

	value *f2sumd;
	cudaMalloc(&f2sumd, sizeof(value));
	void *ts = NULL;
	size_t tsn = 0;
	cub::DeviceReduce::Sum(ts, tsn, v2d, f2sumd, f2->n);
	cudaMalloc(&ts, tsn);
	cub::DeviceReduce::Sum(ts, tsn, v2d, f2sumd, f2->n);
	cudaFree(ts);
	cudaMemcpy(f2sum, f2sumd, sizeof(value), cudaMemcpyDeviceToHost);
	cudaFree(f2sumd);

	value *sepsumd;
	cudaMalloc(&sepsumd, sizeof(value));
	ts = NULL;
	tsn = 0;
	cub::DeviceReduce::Sum(ts, tsn, sd, sepsumd, sep->n);
	cudaMalloc(&ts, tsn);
	cub::DeviceReduce::Sum(ts, tsn, sd, sepsumd, sep->n);
	cudaFree(ts);
	cudaMemcpy(sepsum, sepsumd, sizeof(value), cudaMemcpyDeviceToHost);
	cudaFree(sepsumd);

	gettimeofday(&t1, NULL);
	for (dim i = 0; i < ns; i++) {
		cudaMemcpyAsync(f2->v + i * MAXBLOCKS * n2, v2d + i * MAXBLOCKS * n2, sizeof(value) * n2 * blocks[i], cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(sep->v + i * MAXBLOCKS, sd + i * MAXBLOCKS, sizeof(value) * blocks[i], cudaMemcpyDeviceToHost, streams[i]);
	}

	gettimeofday(&t2, NULL);
	(*transfer) += (t2.tv_usec - t1.tv_usec) / 1e3 + (t2.tv_sec - t1.tv_sec) * 1e3;
	CubDebugExit(cudaPeekAtLastError());
	CubDebugExit(cudaDeviceSynchronize());
	cudaFree(v1d);
	cudaFree(v2d);
	cudaFree(sd);
	free(streams);
	free(blocks);
}
