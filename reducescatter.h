#ifndef REDUCESCATTER_H_
#define REDUCESCATTER_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#ifdef __CUDACC__ // CUDA

#define MAXBLOCKS (1024 * 1024)
//#define ALGORITHM BLOCK_REDUCE_RAKING
//#define ALGORITHM BLOCK_REDUCE_WARP_REDUCTIONS
#define ALGORITHM BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
#define VECTORIZED
#define THREADS 128

#define CUB_STDERR
#include "cub/util_device.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/block/block_reduce.cuh"

#ifdef VECTORIZED
#define LOADDIRECTBLOCKED LoadDirectBlockedVectorized
#define STOREDIRECTBLOCKED StoreDirectBlockedVectorized
#else
#define LOADDIRECTBLOCKED LoadDirectBlocked
#define STOREDIRECTBLOCKED StoreDirectBlocked
#endif

#endif

#include "types.h"
#include "common.h"
#include "preprocess.h"

void updatepotential(func *f1, func *f2, func *sep, const dim *domains, value *f2sum, value *sepsum);

#endif /* REDUCESCATTER_H_ */
