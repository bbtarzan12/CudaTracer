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

#include <freeimage.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int TRACE_SAMPLES = 1000;
constexpr int TRACE_OUTER_LOOP_X = 4;
constexpr int TRACE_OUTER_LOOP_Y = 3;
constexpr float EPSILON = 1e-2;
constexpr float INF = 3.402823466e+38F;
constexpr int MAX_DEPTH = 5;
constexpr int ROULETTE_DEPTH = 3;
constexpr bool ENABLE_SURFACE_ACNE = false;
constexpr int MAX_BUILD_PHOTON_TRHESHOLD = 5;

using namespace glm;

enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC };

int oldTimeSinceStart = 0;
float deltaTime = 0;

// RealTime
GLuint viewGLTexture;
cudaGraphicsResource* viewResource;
cudaArray* viewArray;
bool cudaToggle = true;
bool cudaDirty = false;
int frame = 1;

// HDR Texture
constexpr char* HDR_FILE_NAME = "reading_room_4k.hdr";
constexpr int HDRWidth = 4096;
constexpr int HDRHeight = 2048;
texture<float4, 1, cudaReadModeElementType> HDRtexture;
float4* cudaHDRmap;


// Rendering
bool enableDof = false;
bool enablePhoton = false;
bool enableSaveImage = false;

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

BOOL SwapRedBlue32(FIBITMAP* dib)
{
	if (FreeImage_GetImageType(dib) != FIT_BITMAP)
	{
		return FALSE;
	}

	const unsigned bytesperpixel = FreeImage_GetBPP(dib) / 8;
	if (bytesperpixel > 4 || bytesperpixel < 3)
	{
		return FALSE;
	}

	const unsigned height = FreeImage_GetHeight(dib);
	const unsigned pitch = FreeImage_GetPitch(dib);
	const unsigned lineSize = FreeImage_GetLine(dib);

	BYTE* line = FreeImage_GetBits(dib);
	for (unsigned y = 0; y < height; ++y, line += pitch)
	{
		for (BYTE* pixel = line; pixel < line + lineSize; pixel += bytesperpixel)
		{
			std::swap(pixel[0], pixel[2]);
		}
	}

	return TRUE;
}

#endif
