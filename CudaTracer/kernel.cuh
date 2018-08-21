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

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int TRACE_SAMPLES = 1024;
constexpr auto TRACE_OUTER_LOOP_X = 1;
constexpr auto TRACE_OUTER_LOOP_Y = 1;
constexpr float EPSILON = 1e-10;
constexpr auto INF = 3.402823466e+38F;
constexpr int MAX_DEPTH = 7;
constexpr int ROULETTE_DEPTH = 4;

// RealTime
GLuint viewGLTexture;
cudaGraphicsResource* viewResource;
cudaArray* viewArray;
bool cudaToggle = true;
bool cudaDirty = false;
int frame = 1;

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

#endif
