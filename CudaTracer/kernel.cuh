#ifndef H_CUDAUTILS
#define H_CUDAUTILS

#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <windows.h> 
#include <cuda_gl_interop.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr auto WIDTH = 1280;
constexpr auto HEIGHT = 720;
constexpr auto TRACE_SAMPLES = 50;
constexpr auto EPSILON = 1e-10;
constexpr auto INF = 3.402823466e+38F;
constexpr auto MAX_DEPTH = 1000;
constexpr auto ROULETTE_DEPTH = 5;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ static float GetRandom(unsigned int *seed0, unsigned int *seed1)
{
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	// Convert to float
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

	return (res.f - 2.f) / 2.f;
}


#endif
