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

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <freeimage.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int TRACE_SAMPLES = 10;
constexpr int TRACE_OUTER_LOOP_X = 4;
constexpr int TRACE_OUTER_LOOP_Y = 3;
constexpr float EPSILON = 1e-3f;
constexpr float INF = 3.402823466e+38F;
constexpr int MAX_DEPTH = 5;
constexpr int ROULETTE_DEPTH = 3;
constexpr bool ENABLE_SURFACE_ACNE = false;


using namespace glm;
using namespace std;

enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC };

int oldTimeSinceStart = 0;
float deltaTime = 0;

// RealTime
GLuint viewGLTexture;
cudaGraphicsResource* viewResource;
cudaArray* viewArray;
bool cudaToggle = false;
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
bool enableDirectLighting = true;
constexpr bool ENABLE_SMOOTH_NORMAL = true;

// OpenGL Debug
bool enableDrawNormal = false;
bool enableDrawKDTree = false;

// Photon
constexpr int MAX_BUILD_PHOTON_TRHESHOLD = 5;
constexpr int MAX_PHOTONS = 10000;

// KD Tree
#define ENABLE_KDTREE 1
constexpr int KDTREE_THRESHOLD = 16;
constexpr int KDTREE_MAX_STACK = 1024;
constexpr int KDTREE_MAX_DEPTH = 11;

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

#pragma region KD Tree

template<class T>
class DeviceStack {
public:
	__device__  DeviceStack() {
		ptr = 0;
		//gpuErrorCheck(cudaMalloc((void**)&data, sizeof(T)*GPUKDTREEMAXSTACK));
	}
	__device__  ~DeviceStack() {}//gpuErrorCheck(cudaFree(data);}
	__inline__ __device__  void push(const T& t) { data[ptr++] = t; if (ptr > KDTREE_MAX_STACK)printf("stack over flow!"); }
	__inline__ __device__  T pop() { return data[--ptr]; }
	__inline__ __device__  bool empty() { return ptr <= 0; }
public:
	unsigned int ptr;
	T data[KDTREE_MAX_STACK];
};

template<class T>
class DeviceVector {
public:
	DeviceVector() {}
	~DeviceVector() {
		gpuErrorCheck(cudaFree(data));
		gpuErrorCheck(cudaFree(d_size));
		gpuErrorCheck(cudaFree(d_ptr));
	}
	void allocateMemory(unsigned int n) {
		h_size = n;
		h_ptr = 0;
		gpuErrorCheck(cudaMalloc((void**)&d_size, sizeof(unsigned int)));
		gpuErrorCheck(cudaMalloc((void**)&d_ptr, sizeof(unsigned int)));
		gpuErrorCheck(cudaMemcpy(d_size, &h_size, sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_ptr, &h_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMalloc((void**)&data, sizeof(T)*n));
		thrustPtr = thrust::device_ptr<T>(data);
	}
	void CopyToHost(T* dist) {
		gpuErrorCheck(cudaMemcpy(&h_ptr, d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(dist, data, sizeof(T)*h_ptr, cudaMemcpyDeviceToHost));
	}
	unsigned int size() {
		gpuErrorCheck(cudaMemcpy(&h_ptr, d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return h_ptr;
	}

	__inline__ __device__ static unsigned int push_back(T* d, unsigned int* ptr, T& t) {
		unsigned int i = atomicAdd(ptr, 1);
		d[i] = t;
		return i;
	}
	__inline__ __device__ static void pop(T* d, unsigned int* ptr, T& t) {
		unsigned int i = atomicAdd(ptr, -1);
		t = d[i - 1];
	}
	__inline__ __device__ static bool empty(unsigned int* ptr) {
		if (*ptr <= 0)
			return true;
		return false;
	}
	__host__ bool h_empty() {
		gpuErrorCheck(cudaMemcpy(&h_ptr, d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		if (h_ptr <= 0)
			return true;
		return false;
	}
	__host__ void h_clear() {
		h_ptr = 0;
		gpuErrorCheck(cudaMemcpy(d_ptr, &h_ptr, sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
	__inline__ __device__ static void clear(unsigned int* ptr) {
		*ptr = 0;
	}
	__host__ __device__
		T &operator[](int i) { return data[i]; }

public:
	unsigned int h_size;
	unsigned int h_ptr;
	unsigned int* d_size; // memory size
	unsigned int* d_ptr; // data size
	T* data;
	thrust::device_ptr<T> thrustPtr; // in order to use thrust lib algorithms
};


#pragma endregion KD Tree


#endif