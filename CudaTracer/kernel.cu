#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "kernel.cuh"
#include "Input.h"

#pragma region Structs

struct Ray
{
	vec3 origin;
	vec3 direction;
	__host__ __device__ Ray(vec3 origin, vec3 direction)
	{
		this->origin = origin + direction * (ENABLE_SURFACE_ACNE ? 0 : EPSILON);
		this->direction = direction;
	}
};

struct Photon
{
	__host__ __device__ Photon()
	{
		position = vec3(0, 0, 0);
		normal = vec3(0, 0, 0);
		power = vec3(0, 0, 0);
		type = NONE;
		isHit = false;
	}

	vec3 position;
	vec3 normal;
	vec3 power;
	MaterialType type;
	bool isHit;
};

struct Camera
{
	__host__ __device__ Camera()
	{
		proj = glm::mat4(1.0f);
		position = glm::vec3(0.0f, 46.0f, 126.0f);
		fov = 70.0f;
		nearPlane = 0.1f;
		farPlane = 1000.0f;
		moveSpeed = 25.0f;
		mouseSpeed = 10.0f;
		pitch = 0.0f;
		yaw = 180.0f;
		view = mat4(0);
		proj = mat4(0);
		aperture = 0;
		focalDistance = 0.1f;
	}

	__device__ Ray GetRay(curandState* randState, int x, int y, bool dof)
	{
		float jitterValueX = curand_uniform(randState) - 0.5;
		float jitterValueY = curand_uniform(randState) - 0.5;

		vec3 wDir = glm::normalize(-forward);
		vec3 uDir = glm::normalize(cross(up, wDir));
		vec3 vDir = glm::cross(wDir, -uDir);

		float top = __tanf(fov * glm::pi<float>() / 360.0f);
		float right = aspectRatio * top;
		float bottom = -top;
		float left = -right;

		float imPlaneUPos = left + (right - left)*(((float)x + jitterValueX) / (float)width);
		float imPlaneVPos = bottom + (top - bottom)*(((float)y + jitterValueY) / (float)height);

		vec3 originDirection = imPlaneUPos * uDir + imPlaneVPos * vDir - wDir;
		vec3 pointOnImagePlane = position + ((originDirection) * focalDistance);
		if (dof)
		{
			vec3 aperturePoint = vec3(0, 0, 0);

			if (aperture >= EPSILON)
			{
				float r1 = curand_uniform(randState);
				float r2 = curand_uniform(randState);

				float angle = two_pi<float>() * r1;
				float distance = aperture * sqrt(r2);
				float apertureX = __cosf(angle) * distance;
				float apertureY = __sinf(angle) * distance;

				aperturePoint = position + (wDir * apertureX) + (uDir * apertureY);
			}
			else
			{
				aperturePoint = position;
			}
			return Ray(aperturePoint, normalize(pointOnImagePlane - aperturePoint));
		}
		else
		{
			return Ray(position, normalize(originDirection));
		}
	}

	void UpdateScreen(int width, int height)
	{
		this->width = width;
		this->height = height;
		this->aspectRatio = width / (float)height;

		glViewport(0, 0, width, height);
		proj = perspective(radians(fov), aspectRatio, nearPlane, farPlane);
	}

	void UpdateCamera(float deltaTime)
	{
		vec2 input = vec2(IsKeyDown('w') ? 1 : IsKeyDown('s') ? -1 : 0, IsKeyDown('d') ? 1 : IsKeyDown('a') ? -1 : 0);
		if (IsMouseDown(1))
			HandleRotate(deltaTime);
		HandleMove(input, deltaTime);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fov, aspectRatio, nearPlane, farPlane);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		forward.x = cos(radians(pitch)) * sin(radians(yaw));
		forward.y = sin(radians(pitch));
		forward.z = cos(radians(pitch)) * cos(radians(yaw));
		forward = normalize(forward);
		right = normalize(cross(forward, vec3(0, 1, 0)));
		up = normalize(cross(right, forward));

		mat4 viewMatrix = lookAt(position, position + forward, up);
		if (view != viewMatrix)
		{
			cudaDirty = true;
			view = viewMatrix;
		}
		glMultMatrixf(value_ptr(view));
		toggleMouseMovement = IsMouseDown(1);
	}

	bool toggleMouseMovement;
	float width, height;
	float moveSpeed, mouseSpeed;
	float nearPlane, farPlane;
	float fov;
	float aspectRatio;
	float pitch, yaw;

	// fov
	float aperture, focalDistance;

	vec3 position;
	vec3 forward, up, right;

	mat4 view;
	mat4 proj;

private:
	void HandleRotate(float deltaTime)
	{
		if (toggleMouseMovement == false)
		{
			WarpMouse(width / 2, height / 2);
			return;
		}

		int xPos, yPos;
		GetMousePos(xPos, yPos);

		pitch += mouseSpeed * float(height / 2 - yPos) * deltaTime;
		yaw += mouseSpeed * float(width / 2 - xPos) * deltaTime;

		pitch = clamp(pitch, -89.0f, 89.0f);
		yaw = mod(yaw, 360.0f);

		WarpMouse(width / 2, height / 2);
	}

	void HandleMove(vec2 input, float deltaTime)
	{
		position += (forward * input.x + right * input.y) * deltaTime * moveSpeed;
	}
};

struct Material
{
	__host__ __device__ Material(MaterialType type = DIFF, vec3 color = vec3(0), vec3 emission = vec3(0))
	{
		this->type = type;
		this->color = color;
		this->emission = emission;
	}
	MaterialType type;
	vec3 color;
	vec3 emission;
};

struct ObjectIntersection
{
	__host__ __device__ ObjectIntersection(bool hit = false, float t = 0, vec3 normal = vec3(0), Material material = Material())
	{
		this->hit = hit;
		this->t = t;
		this->normal = normal;
		this->material = material;
		hitPtr = nullptr;
	}
	bool hit;
	float t;
	vec3 normal;
	Material material;
	void* hitPtr;
};

struct Triangle
{
	__host__ __device__ Triangle() {}
	__host__ __device__ Triangle(vec3 pos0, vec3 pos1, vec3 pos2, vec3 nor0, vec3 nor1, vec3 nor2, Material material)
	{
		pos[0] = pos0;
		pos[1] = pos1;
		pos[2] = pos2;
		nor[0] = normalize(nor0);
		nor[1] = normalize(nor1);
		nor[2] = normalize(nor2);
		hasTexture = false;
		this->material = material;
	}

	__host__ __device__ Triangle(vec3 pos0, vec3 pos1, vec3 pos2, vec3 nor0, vec3 nor1, vec3 nor2, vec3 tex0, vec3 tex1, vec3 tex2, Material material)
	{
		pos[0] = pos0;
		pos[1] = pos1;
		pos[2] = pos2;
		nor[0] = normalize(nor0);
		nor[1] = normalize(nor1);
		nor[2] = normalize(nor2);
		tex[0] = tex0;
		tex[1] = tex1;
		tex[2] = tex2;
		hasTexture = true;
		this->material = material;
	}

	__device__ ObjectIntersection Intersect(const Ray &ray, vec3 position) const
	{
		bool hit = false;
		float u, v, t = 0;
		vec3 pos[3] = { this->pos[0] + position, this->pos[1] + position, this->pos[2] + position };

		vec3 normal = vec3(0); /*= normalize(cross(pos[1] - pos[0], pos[2] - pos[0]));*/

		vec3 v0v1 = pos[1] - pos[0];
		vec3 v0v2 = pos[2] - pos[0];
		vec3 pvec = cross(ray.direction, v0v2);
		float det = dot(v0v1, pvec);
		if (fabs(det) < EPSILON) return ObjectIntersection(hit, t, normal, material);

		float invDet = 1.0f / det;
		vec3 tvec = ray.origin - pos[0];
		u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) return ObjectIntersection(hit, t, normal, material);

		vec3 qvec = cross(tvec, v0v1);
		v = dot(ray.direction, qvec) * invDet;
		if (v < 0 || u + v > 1) return ObjectIntersection(hit, t, normal, material);

		t = dot(v0v2, qvec) * invDet;

		if (t < EPSILON) return ObjectIntersection(hit, t, normal, material);
		if (ENABLE_SMOOTH_NORMAL)
			normal = normalize((1 - u - v) * nor[0] + u * nor[1] + v * nor[2]);
		else
			normal = normalize(cross(v0v1, v0v2));
		hit = true;
		return ObjectIntersection(hit, t, normal, material);
	}

	vec3 pos[3];
	vec3 nor[3];
	vec3 tex[3];
	Material material;
	bool hasTexture;
};

struct Sphere
{
	__host__ __device__ Sphere(vec3 position = vec3(0), float radius = 0, Material material = Material())
	{
		this->position = position;
		this->radius = radius;
		this->material = material;
	}
	float radius;
	vec3 position;
	Material material;
	__device__ ObjectIntersection Intersect(const Ray &ray)
	{
		bool hit = false;
		float distance = 0, t = 0;
		vec3 normal = vec3(0, 0, 0);
		vec3 op = position - ray.origin;
		float b = dot(op, ray.direction);
		float det = b * b - dot(op, op) + radius * radius;

		if (det < EPSILON)
			return ObjectIntersection(hit, t, normal, material);
		else
			det = glm::sqrt(det);

		distance = (t = b - det) > EPSILON ? t : ((t = b + det) > EPSILON ? t : 0);
		if (distance > EPSILON)
		{
			hit = true;
			normal = normalize(ray.direction * distance - op);
		}
		ObjectIntersection result = ObjectIntersection(hit, distance, normal, material);
		if (hit == true)
			result.hitPtr = this;
		return result;
	}
	__device__ vec3 RandomPoint(curandState* randState)
	{
		float theta = curand_uniform(randState) * pi<float>();
		float phi = curand_uniform(randState) * two_pi<float>();
		// Convert to Cartesian and scale by radius
		float dxr = radius * sin(theta) * cos(phi);
		float dyr = radius * sin(theta) * sin(phi);
		float dzr = radius * cos(theta);
		return vec3(position.x + dxr, position.y + dyr, position.z + dzr);
	}
};

#pragma region KDTree

struct AABB
{
	__device__ __host__ AABB()
	{
		bounds[0] = vec3(0);
		bounds[1] = vec3(1);
	}
	__device__ __host__ AABB(vec3 min, vec3 max)
	{
		bounds[0] = min;
		bounds[1] = max;
	}
	__device__ __host__ AABB(Triangle* triangles, int count)
	{
		for (int i = 0; i < count; i++)
		{
			Expand(triangles[i]);
		}
	}

	__device__ __host__ void Expand(Triangle triangle)
	{
		Expand
		(
			vec3
			(
				min(min(triangle.pos[0].x, triangle.pos[1].x), triangle.pos[2].x),
				min(min(triangle.pos[0].y, triangle.pos[1].y), triangle.pos[2].y),
				min(min(triangle.pos[0].z, triangle.pos[1].z), triangle.pos[2].z)
			),
			vec3
			(
				max(max(triangle.pos[0].x, triangle.pos[1].x), triangle.pos[2].x),
				max(max(triangle.pos[0].y, triangle.pos[1].y), triangle.pos[2].y),
				max(max(triangle.pos[0].z, triangle.pos[1].z), triangle.pos[2].z)
			)
		);
	}

	__device__ __host__ void Expand(vec3 min, vec3 max)
	{
		if (min.x < bounds[0].x) bounds[0].x = min.x;
		if (min.y < bounds[0].y) bounds[0].y = min.y;
		if (min.z < bounds[0].z) bounds[0].z = min.z;

		if (max.x > bounds[1].x) bounds[1].x = max.x;
		if (max.y > bounds[1].y) bounds[1].y = max.y;
		if (max.z > bounds[1].z) bounds[1].z = max.z;
	}
	vec3 bounds[2];
};
struct KDTreeNode
{
	__device__ __host__ KDTreeNode(int l = -1, int r = -1, int sa = -1, int ti = 0, int tn = 0, float sp = 0, int d = 0)
	{
		leftChild = l; rightChild = r; splitAxis = sa; triangleIndex = ti; triangleNumber = tn; splitPos = sp; depth = d;
	}
	__device__ __host__ KDTreeNode(const KDTreeNode& g)
	{
		leftChild = g.leftChild; rightChild = g.rightChild; splitAxis = g.splitAxis; triangleIndex = g.triangleIndex;
		triangleNumber = g.triangleNumber; splitPos = g.splitPos; nodeAABB = g.nodeAABB; depth = g.depth;
	}
	int leftChild;
	int rightChild;
	int splitAxis;
	int triangleIndex;
	int triangleNumber;
	float splitPos;
	int depth;
	AABB nodeAABB;
};

__device__ void AABBMax(vec3* x, vec3* y, vec3* z, vec3* dist)
{
	float xmax = x->x > y->x ? x->x : y->x;
	xmax = xmax > z->x ? xmax : z->x;
	float ymax = x->y > y->y ? x->y : y->y;
	ymax = ymax > z->y ? ymax : z->y;
	float zmax = x->z > y->z ? x->z : y->z;
	zmax = zmax > z->z ? zmax : z->z;
	dist->x = xmax;
	dist->y = ymax;
	dist->z = zmax;
}
__device__ void AABBMin(vec3* x, vec3* y, vec3* z, vec3* dist)
{
	float xmax = x->x < y->x ? x->x : y->x;
	xmax = xmax < z->x ? xmax : z->x;
	float ymax = x->y < y->y ? x->y : y->y;
	ymax = ymax < z->y ? ymax : z->y;
	float zmax = x->z < y->z ? x->z : y->z;
	zmax = zmax < z->z ? zmax : z->z;
	dist->x = xmax;
	dist->y = ymax;
	dist->z = zmax;
}
__global__ void CreateAABB(int n, Triangle* tri, AABB* aabb)
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= n)
		return;
	AABBMax(&(tri[tid].pos[0]), &(tri[tid].pos[1]), &(tri[tid].pos[2]), &(aabb[tid].bounds[1]));
	AABBMin(&(tri[tid].pos[0]), &(tri[tid].pos[1]), &(tri[tid].pos[2]), &(aabb[tid].bounds[0]));
}
__global__ void InitRoot(int nTri, KDTreeNode* nodes, unsigned int* nodesPtr, int* activeList, unsigned int* activeListPtr, unsigned int* nextListPtr, unsigned int* smallListPtr, unsigned int* tnaPtr, AABB aabb)
{
	DeviceVector<int>::clear(activeListPtr);
	DeviceVector<int>::clear(nextListPtr);
	DeviceVector<int>::clear(smallListPtr);
	DeviceVector<int>::clear(tnaPtr);
	DeviceVector<KDTreeNode>::clear(nodesPtr);

	KDTreeNode n;
	n.triangleIndex = 0;
	n.triangleNumber = nTri;
	n.nodeAABB = aabb;
	n.depth = 0;
	DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, n);
	*(tnaPtr) = nTri;

	int i = 0;
	DeviceVector<int>::push_back(activeList, activeListPtr, i);
}
__global__ void CopyTriangle(int* tna, int n)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= n)
		return;
	tna[tid] = tid;
}
__global__ void MidSplitNode(Triangle* tri, AABB* aabb, int nTri, KDTreeNode* nodes, unsigned int* nodesPtr, int* activeList, unsigned int* activeListPtr, int* nextList, unsigned int* nextListPtr, int* smallList, unsigned int* smallListPtr, int* tna, unsigned int* tnaPtr, int* tnahelper, unsigned int* tnahelperPtr, unsigned int tnaStartPtr)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= *activeListPtr)
		return;
	//printf("tid=%d\n",tid);
	int id = activeList[tid];
	//printf("node triangle number=%d\n",nodes[id].triangleNumber);
	int leftid;
	int rightid;
	float sp;

	if (nodes[id].depth > KDTREE_MAX_DEPTH)
		return;

	//KDTreeNode currentNode(nodes[id]);
	vec3 volume = nodes[id].nodeAABB.bounds[1] - nodes[id].nodeAABB.bounds[0];
	if (volume.x >= volume.y && volume.x >= volume.z)// split x
	{
		nodes[id].splitAxis = 0;
		sp = nodes[id].nodeAABB.bounds[0].x + volume.x / 2.0f;
		nodes[id].splitPos = sp;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].x = sp;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].x = nodes[id].nodeAABB.bounds[1].x;
		atarashiiNode.nodeAABB.bounds[0].x = sp;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	else if (volume.y >= volume.x && volume.y >= volume.z)// split y
	{
		nodes[id].splitAxis = 1;
		sp = nodes[id].nodeAABB.bounds[0].y + volume.y / 2.0f;
		nodes[id].splitPos = sp;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].y = sp;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].y = nodes[id].nodeAABB.bounds[1].y;
		atarashiiNode.nodeAABB.bounds[0].y = sp;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	else // split z
	{
		nodes[id].splitAxis = 2;
		sp = nodes[id].nodeAABB.bounds[0].z + volume.z / 2.0f;
		nodes[id].splitPos = sp;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].z = sp;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].z = nodes[id].nodeAABB.bounds[1].z;
		atarashiiNode.nodeAABB.bounds[0].z = sp;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	// split triangles
	int leftcount = 0;
	int rightcount = 0;
	unsigned int tnapos;
	int endPtr = nodes[id].triangleIndex + nodes[id].triangleNumber - 1;
	/*printf("triangleIndex=%d\n", currentNode.triangleIndex);
	printf("triangleNumber=%d\n", currentNode.triangleNumber);
	printf("endPtr=%d\n", endPtr);*/
	for (int i = nodes[id].triangleIndex; i <= endPtr; i++)
	{
		int triid = tna[i];

		switch (nodes[id].splitAxis)
		{
		case 0:
			if (aabb[triid].bounds[0].x <= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].x >= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		case 1:
			if (aabb[triid].bounds[0].y <= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].y >= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		case 2:
			if (aabb[triid].bounds[0].z <= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].z >= sp) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		}
	}
	//printf("leftcount=%d\nrightcount=%d\n", leftcount, rightcount);
	nodes[leftid].triangleNumber = leftcount;
	nodes[rightid].triangleNumber = rightcount;

	nodes[leftid].depth = nodes[id].depth + 1;
	nodes[rightid].depth = nodes[id].depth + 1;

	//printf("node %d was splited with left = %d and right = %d with sp=%.5f tna=%d\n", id, leftcount, rightcount, sp, *tnaPtr);
	// add to nextList
	if (leftcount > KDTREE_THRESHOLD * 2)
		DeviceVector<int>::push_back(nextList, nextListPtr, leftid);
	else if (leftcount > KDTREE_THRESHOLD)
		DeviceVector<int>::push_back(smallList, smallListPtr, leftid);
	if (rightcount > KDTREE_THRESHOLD * 2)
		DeviceVector<int>::push_back(nextList, nextListPtr, rightid);
	else if (rightcount > KDTREE_THRESHOLD)
		DeviceVector<int>::push_back(smallList, smallListPtr, rightid);
}
__global__ void SAHSplitNode(Triangle* tri, AABB* aabb, int nTri, KDTreeNode* nodes, unsigned int* nodesPtr, int* smallList, unsigned int* smallListPtr, int* nextList, unsigned int* nextListPtr, int* tna, unsigned int* tnaPtr, int* tnahelper, unsigned int* tnahelperPtr, unsigned int tnaStartPtr)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= *smallListPtr)
		return;
	//printf("tid=%d\n",tid);
	int id = smallList[tid];
	//printf("node triangle number=%d\n",nodes[id].triangleNumber);
	int leftid;
	int rightid;
	float tpos;
	//KDTreeNode currentNode(nodes[id]);
	if (nodes[id].depth > KDTREE_MAX_DEPTH)
		return;
	vec3 volume = nodes[id].nodeAABB.bounds[1] - nodes[id].nodeAABB.bounds[0];
	if (volume.x >= volume.y && volume.x >= volume.z)// split x
	{
		nodes[id].splitAxis = 0;
		// looking for best candidate
		float minsah = 999999.0f;
		float minpos;

		for (float p = 0.1f; p < 1.0f; p += 0.1f) {
			tpos = nodes[id].nodeAABB.bounds[0].x + volume.x*p;
			int ct1, ct2;
			ct1 = ct2 = 0;
			for (int i = nodes[id].triangleIndex, j = 0; j < nodes[id].triangleNumber; i++, j++) {
				if ((aabb[tnaPtr[i]].bounds[0].x + aabb[tnaPtr[i]].bounds[1].x) / 2 < tpos)
					ct1++;
				else
					ct2++;
			}
			float sah = ct1 * p + ct2 * (1 - p);
			if (sah < minsah) {
				minsah = sah;
				minpos = tpos;
			}
		}
		nodes[id].splitPos = tpos;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].x = tpos;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].x = nodes[id].nodeAABB.bounds[1].x;
		atarashiiNode.nodeAABB.bounds[0].x = tpos;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	else if (volume.y >= volume.x && volume.y >= volume.z)// split y
	{
		nodes[id].splitAxis = 1;
		// looking for best candidate
		float minsah = 999999.0f;
		float minpos;

		for (float p = 0.1f; p < 1.0f; p += 0.1f) {
			tpos = nodes[id].nodeAABB.bounds[0].y + volume.y*p;
			int ct1, ct2;
			ct1 = ct2 = 0;
			for (int i = nodes[id].triangleIndex, j = 0; j < nodes[id].triangleNumber; i++, j++) {
				if ((aabb[tnaPtr[i]].bounds[0].y + aabb[tnaPtr[i]].bounds[1].y) / 2 < tpos)
					ct1++;
				else
					ct2++;
			}
			float sah = ct1 * p + ct2 * (1 - p);
			if (sah < minsah) {
				minsah = sah;
				minpos = tpos;
			}
		}
		nodes[id].splitPos = tpos;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].y = tpos;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].y = nodes[id].nodeAABB.bounds[1].y;
		atarashiiNode.nodeAABB.bounds[0].y = tpos;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	else // split z
	{
		nodes[id].splitAxis = 2;
		// looking for best candidate
		float minsah = 999999.0f;
		float minpos;

		for (float p = 0.1f; p < 1.0f; p += 0.1f) {
			tpos = nodes[id].nodeAABB.bounds[0].z + volume.z*p;
			int ct1, ct2;
			ct1 = ct2 = 0;
			for (int i = nodes[id].triangleIndex, j = 0; j < nodes[id].triangleNumber; i++, j++) {
				if ((aabb[tnaPtr[i]].bounds[0].z + aabb[tnaPtr[i]].bounds[1].z) / 2 < tpos)
					ct1++;
				else
					ct2++;
			}
			float sah = ct1 * p + ct2 * (1 - p);
			if (sah < minsah) {
				minsah = sah;
				minpos = tpos;
			}
		}
		nodes[id].splitPos = tpos;

		KDTreeNode atarashiiNode;
		atarashiiNode.nodeAABB = nodes[id].nodeAABB;
		atarashiiNode.nodeAABB.bounds[1].z = tpos;
		leftid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].leftChild = leftid;

		atarashiiNode.nodeAABB.bounds[1].z = nodes[id].nodeAABB.bounds[1].z;
		atarashiiNode.nodeAABB.bounds[0].z = tpos;
		rightid = DeviceVector<KDTreeNode>::push_back(nodes, nodesPtr, atarashiiNode);
		nodes[id].rightChild = rightid;
	}
	//printf("sp=%.3f\n",sp);
	// split triangles
	int leftcount = 0;
	int rightcount = 0;
	unsigned int tnapos;
	int endPtr = nodes[id].triangleIndex + nodes[id].triangleNumber - 1;
	/*printf("triangleIndex=%d\n", currentNode.triangleIndex);
	printf("triangleNumber=%d\n", currentNode.triangleNumber);
	printf("endPtr=%d\n", endPtr);*/
	for (int i = nodes[id].triangleIndex; i <= endPtr; i++)
	{
		int triid = tna[i];

		switch (nodes[id].splitAxis)
		{
		case 0:
			if (aabb[triid].bounds[0].x <= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				//DeviceVector<int>::push_back(tnahelper, tnahelperPtr, leftid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].x >= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		case 1:
			if (aabb[triid].bounds[0].y <= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].y >= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		case 2:
			if (aabb[triid].bounds[0].z <= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = leftid;
				leftcount++;
			}
			if (aabb[triid].bounds[1].z >= tpos) {
				tnapos = DeviceVector<int>::push_back(tna, tnaPtr, triid);
				tnahelper[tnapos - tnaStartPtr] = rightid;
				rightcount++;
			}
			break;
		}
	}
	//printf("leftcount=%d\nrightcount=%d\n", leftcount, rightcount);
	nodes[leftid].triangleNumber = leftcount;
	nodes[rightid].triangleNumber = rightcount;
	//printf("node %d was splited with left = %d and right = %d with tna=%d\n", id, leftcount, rightcount, *tnaPtr);
	// add to nextList

	nodes[leftid].depth = nodes[id].depth+1;
	nodes[rightid].depth = nodes[id].depth+1;

	if (leftcount > KDTREE_THRESHOLD)
		DeviceVector<int>::push_back(smallList, smallListPtr, leftid);

	if (rightcount > KDTREE_THRESHOLD)
		DeviceVector<int>::push_back(smallList, smallListPtr, rightid);
}
__global__ void CalculateTriangleIndex(int start, int end, int base, KDTreeNode* nodes)
{
	int count = 0;
	int basecount = nodes[base].triangleIndex + nodes[base].triangleNumber;
	for (int i = start; i <= end; i++)
	{
		nodes[i].triangleIndex = basecount + count;
		count += nodes[i].triangleNumber;
	}
}
__device__ float KDRayTraversal(KDTreeNode* root, Ray ray, float& minDist, float& distance, vec3 position)
{
	if (root->triangleNumber <= 0)
		return;

	vec3 minBox = root->nodeAABB.bounds[0] + position;
	vec3 maxBox = root->nodeAABB.bounds[1] + position;

	if (ray.direction.x < 0)
	{
		ray.origin.x = minBox.x + maxBox.x - ray.origin.x;
		ray.direction.x = -ray.direction.x;
	}
	if (ray.direction.y < 0)
	{
		ray.origin.y = minBox.y + maxBox.y - ray.origin.y;
		ray.direction.y = -ray.direction.y;
	}
	if (ray.direction.z < 0)
	{
		ray.origin.z = minBox.z + maxBox.z - ray.origin.z;
		ray.direction.z = -ray.direction.z;
	}

	vec3 div = 1.0f / ray.direction;
	vec3 tMin = (minBox - ray.origin) * div;
	vec3 tMax = (maxBox - ray.origin) * div;

	float tmin = max(max(tMin.x, tMin.y), tMin.z);
	float tmax = min(min(tMax.x, tMax.y), tMax.z);

	if (tmin <= tmax)
	{
		if (tmin < minDist)
		{
			distance = tmin;
			return true;
		}
		else
			return false;
	}
	else
		return false;

}
__device__ ObjectIntersection RayKDTreeTraversal(KDTreeNode* nodes, int* tna, Ray ray, Triangle* triangles, vec3 position)
{
	int currentid = 0, leftid = 0, rightid = 0, cid = 0;
	bool isHit = false;
	float minDist = INF;
	vec3 normal = vec3(0);
	Material material;
	DeviceStack<int> treestack;
	treestack.push(0);

	float distance = -1.0f;
	vec3 point;

	while (!treestack.empty())
	{
		currentid = treestack.pop();

		//test node intersection
		if (KDRayTraversal(&nodes[currentid], ray, minDist, distance, position))
		{
			leftid = nodes[currentid].leftChild;
			rightid = nodes[currentid].rightChild;

			//// leaf node
			if (leftid == -1)
			{
				for (int i = nodes[currentid].triangleIndex; i < nodes[currentid].triangleIndex + nodes[currentid].triangleNumber; i++)
				{
					ObjectIntersection intersection = triangles[tna[i]].Intersect(ray, position);
					if (intersection.hit && intersection.t < minDist)
					{
						minDist = intersection.t;
						isHit = true;
						normal = intersection.normal;
						material = intersection.material;
					}
				}

				continue;
			}

			// middle node
			if (leftid != -1)
			{
				point = ray.origin + ray.direction * distance;

				if (nodes[currentid].splitAxis == 0)
				{
					if (point.x < nodes[currentid].nodeAABB.bounds[0].x + nodes[currentid].splitPos)
					{
						treestack.push(leftid);
						treestack.push(rightid);
					}
					else
					{
						treestack.push(rightid);
						treestack.push(leftid);
					}

				}
				else if (nodes[currentid].splitAxis == 1)
				{
					if (point.y < nodes[currentid].nodeAABB.bounds[0].y + nodes[currentid].splitPos)
					{
						treestack.push(leftid);
						treestack.push(rightid);
					}
					else
					{
						treestack.push(rightid);
						treestack.push(leftid);
					}

				}
				else if (nodes[currentid].splitAxis == 2)
				{
					if (point.z < nodes[currentid].nodeAABB.bounds[0].z + nodes[currentid].splitPos)
					{
						treestack.push(leftid);
						treestack.push(rightid);
					}
					else
					{
						treestack.push(rightid);
						treestack.push(leftid);
					}

				}
			}
		}
	}
	return ObjectIntersection(isHit, minDist, normal, material);
}

struct MaxX
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[1].x;
	}
};
struct MaxY
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[1].y;
	}
};
struct MaxZ
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[1].z;
	}
};
struct MinX
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[0].x;
	}
};
struct MinY
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[0].y;
	}
};
struct MinZ
{
	__host__ __device__ float operator()(AABB const& x)const {
		return x.bounds[0].z;
	}
};
struct KDTree
{
	KDTree(){}
	KDTree(Triangle* tri, int n)
	{
		h_Triangles = tri;
		nTriangle = n;
		rootAABB = AABB(h_Triangles, nTriangle);
		printf("Root AABB Size : Min : %f %f %f | Max : %f %f %f\n", rootAABB.bounds[0].x, rootAABB.bounds[0].y, rootAABB.bounds[0].z, rootAABB.bounds[1].x, rootAABB.bounds[1].y, rootAABB.bounds[1].z);
	}
	~KDTree() { freeMemory(); }
	void Build()
	{
		int blocksize = (nTriangle + 255) / 256;

		allocateMemory();

		cout << "memcpy on gpu" << endl;
		// calculate AABB
		CreateAABB << <blocksize, 256 >> > (nTriangle, d_Triangles, d_AABB);

		MidSplit();
		SAHSplit();

		cout << "gpu kdtree debug info:" << endl;
		cout << nodes.size() << endl;
		cout << triangleNodeAssociation.size() << endl;
	}

	AABB rootAABB;
	int nTriangle;
	Triangle* d_Triangles;
	Triangle* h_Triangles;
	AABB* d_AABB;

	DeviceVector<KDTreeNode> nodes;
	DeviceVector<int> triangleNodeAssociation;
	DeviceVector<int> triangleNodeAssociationHelper;
	DeviceVector<int> activeList;
	DeviceVector<int> nextList;
	DeviceVector<int> smallList;

private:
	void allocateMemory()
	{
		gpuErrorCheck(cudaMalloc((void**)&d_Triangles, sizeof(Triangle)*nTriangle));
		gpuErrorCheck(cudaMalloc((void**)&d_AABB, sizeof(AABB)*nTriangle));
		gpuErrorCheck(cudaMemcpy(d_Triangles, h_Triangles, sizeof(Triangle)*nTriangle, cudaMemcpyHostToDevice));
		
		nodes.allocateMemory(nTriangle / 3);
		triangleNodeAssociation.allocateMemory(nTriangle * 30);
		triangleNodeAssociationHelper.allocateMemory(nTriangle * 10);
		activeList.allocateMemory(nTriangle / 3);
		nextList.allocateMemory(nTriangle / 3);
		smallList.allocateMemory(nTriangle / 3);
	}
	void freeMemory()
	{
		printf("KD Tree Free\n");
		gpuErrorCheck(cudaFree(d_Triangles));
		gpuErrorCheck(cudaFree(d_AABB));
	}
	AABB CalculateRootAABB()
	{
		thrust::device_ptr<AABB> thrustPtr(d_AABB);
		float maxx = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MaxX(), 0, thrust::maximum<float>());
		float maxy = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MaxY(), 0, thrust::maximum<float>());
		float maxz = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MaxZ(), 0, thrust::maximum<float>());
		float minx = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MinX(), 0, thrust::minimum<float>());
		float miny = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MinY(), 0, thrust::minimum<float>());
		float minz = thrust::transform_reduce(thrustPtr, thrustPtr + nTriangle, MinZ(), 0, thrust::minimum<float>());
		gpuErrorCheck(cudaDeviceSynchronize());

		AABB tmp;

		tmp.bounds[0] = vec3(minx, miny, minz);
		tmp.bounds[1] = vec3(maxx, maxy, maxz);

		return tmp;
	}
	void MidSplit()
	{
		InitRoot << <1, 1 >> > (nTriangle, nodes.data, nodes.d_ptr, activeList.data, activeList.d_ptr, nextList.d_ptr, smallList.d_ptr, triangleNodeAssociation.d_ptr, rootAABB);
		gpuErrorCheck(cudaDeviceSynchronize());

		CopyTriangle << <(nTriangle + 255) / 256, 256 >> > (triangleNodeAssociation.data, nTriangle);
		gpuErrorCheck(cudaDeviceSynchronize());

		while (!activeList.h_empty())
		{
			int base = nodes.size() - 1;
			int startnode = nodes.size();
			int start = triangleNodeAssociation.size();
			triangleNodeAssociationHelper.h_clear();
			MidSplitNode << <(activeList.size() + 255) / 256, 256 >> > (d_Triangles, d_AABB, nTriangle,
				nodes.data,
				nodes.d_ptr,
				activeList.data,
				activeList.d_ptr,
				nextList.data,
				nextList.d_ptr,
				smallList.data,
				smallList.d_ptr,
				triangleNodeAssociation.data,
				triangleNodeAssociation.d_ptr,
				triangleNodeAssociationHelper.data,
				triangleNodeAssociationHelper.d_ptr,
				start);
			gpuErrorCheck(cudaDeviceSynchronize());
			int end = triangleNodeAssociation.size();
			int endnode = nodes.size() - 1;
			int noftna = end - start;
			thrust::sort_by_key(triangleNodeAssociationHelper.thrustPtr, triangleNodeAssociationHelper.thrustPtr + noftna, triangleNodeAssociation.thrustPtr + start);
			gpuErrorCheck(cudaDeviceSynchronize());
			// calculate triangleIndex
			CalculateTriangleIndex << <1, 1 >> > (startnode, endnode, base, nodes.data);
			gpuErrorCheck(cudaDeviceSynchronize());
			// switch aciveList and nextList
			//cout<<"nextlist size:"<<nextList.size()<<" tnasize="<<noftna<<endl;
			gpuErrorCheck(cudaMemcpy(activeList.data, nextList.data, sizeof(int)*nextList.size(), cudaMemcpyDeviceToDevice));
			gpuErrorCheck(cudaMemcpy(activeList.d_ptr, nextList.d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			nextList.h_clear();
			triangleNodeAssociationHelper.h_clear();
			gpuErrorCheck(cudaDeviceSynchronize());
		}
	}
	void SAHSplit()
	{
		{
			while (!smallList.h_empty())
			{
				int base = nodes.size() - 1;
				int startnode = nodes.size();
				int start = triangleNodeAssociation.size();
				triangleNodeAssociationHelper.h_clear();
				SAHSplitNode << <(smallList.size() + 255) / 256, 256 >> > (d_Triangles, d_AABB, nTriangle,
					nodes.data,
					nodes.d_ptr,
					smallList.data,
					smallList.d_ptr,
					nextList.data,
					nextList.d_ptr,
					triangleNodeAssociation.data,
					triangleNodeAssociation.d_ptr,
					triangleNodeAssociationHelper.data,
					triangleNodeAssociationHelper.d_ptr,
					start);
				gpuErrorCheck(cudaDeviceSynchronize());
				int end = triangleNodeAssociation.size();
				int endnode = nodes.size() - 1;
				int noftna = end - start;
				thrust::sort_by_key(triangleNodeAssociationHelper.thrustPtr, triangleNodeAssociationHelper.thrustPtr + noftna, triangleNodeAssociation.thrustPtr + start);
				gpuErrorCheck(cudaDeviceSynchronize());
				// calculate triangleIndex
				CalculateTriangleIndex << <1, 1 >> > (startnode, endnode, base, nodes.data);
				gpuErrorCheck(cudaDeviceSynchronize());
				// switch aciveList and nextList
				//cout<<"nextlist size:"<<nextList.size()<<" tnasize="<<noftna<<endl;
				gpuErrorCheck(cudaMemcpy(smallList.data, nextList.data, sizeof(int)*nextList.size(), cudaMemcpyDeviceToDevice));
				gpuErrorCheck(cudaMemcpy(smallList.d_ptr, nextList.d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToDevice));

				nextList.h_clear();
				triangleNodeAssociationHelper.h_clear();
				gpuErrorCheck(cudaDeviceSynchronize());
			}
		}
	}
};
#pragma endregion KDTree

struct Mesh
{
	__host__ __device__ Mesh() {}
	__host__  Mesh(vec3 position, const char* fileName = "", Material material = Material())
	{
		this->position = position;

		std::string mtlBasePath;
		std::string inputFile = fileName;
		unsigned long pos = inputFile.find_last_of("/");
		mtlBasePath = inputFile.substr(0, pos + 1);

		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> obj_shapes;
		std::vector<tinyobj::material_t> obj_materials;
		std::vector<Material> materials;

		printf("Loading %s...\n", fileName);
		std::string err;
		bool ret = tinyobj::LoadObj(&attrib, &obj_shapes, &obj_materials, &err, inputFile.c_str(), mtlBasePath.c_str());

		if (!err.empty())
			std::cerr << err << std::endl;

		if (!ret) exit(1);

		for (auto & obj_material : obj_materials)
		{
			std::string texturePath = "";

			vec3 diffuseColor = vec3(obj_material.diffuse[0], obj_material.diffuse[1], obj_material.diffuse[2]);
			vec3 emissionColor = vec3(obj_material.emission[0], obj_material.emission[1], obj_material.emission[2]);

			if (!obj_material.diffuse_texname.empty())
			{
				if (obj_material.diffuse_texname[0] == '/') texturePath = obj_material.diffuse_texname;
				texturePath = mtlBasePath + obj_material.diffuse_texname;
				materials.push_back(Material(material.type, diffuseColor, emissionColor));
			}
			else
			{
				materials.push_back(Material(material.type, diffuseColor, emissionColor));
			}
		}

		long shapeSize, faceSize;
		shapeSize = obj_shapes.size();
		std::vector<Triangle>* triangles = new std::vector<Triangle>;

		for (int i = 0; i < shapeSize; i++)
		{
			size_t index_offset = 0;
			faceSize = obj_shapes[i].mesh.num_face_vertices.size();
			for (size_t f = 0; f < faceSize; f++)
			{
				size_t fnum = obj_shapes[i].mesh.num_face_vertices[f];
				vec3 pos[3];
				vec3 nor[3];

				for (int k = 0; k < 3; k++)
				{
					tinyobj::index_t idx = obj_shapes[i].mesh.indices[index_offset + k];
					pos[k] = vec3(
						attrib.vertices[3 * idx.vertex_index + 0],
						attrib.vertices[3 * idx.vertex_index + 1],
						attrib.vertices[3 * idx.vertex_index + 2]
					);
					nor[k] = vec3(
						attrib.normals[3 * idx.normal_index + 0],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]
					);
					nor[k] = normalize(nor[k]);
				}

				vec3 t0_, t1_, t2_;

				//if (obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1 < obj_shapes[i].mesh.texcoords.size())
				//{
				//	t0_ = vec3(
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2],
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2 + 1],
				//		0
				//	);

				//	t1_ = vec3(
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2],
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2 + 1],
				//		0
				//	);

				//	t2_ = vec3(
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2],
				//		obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1],
				//		0
				//	);
				//}
				//else
				//{
				//	t0_ = vec3(0, 0, 0);
				//	t1_ = vec3(0, 0, 0);
				//	t2_ = vec3(0, 0, 0);
				//}

				Triangle triangle;
				if (obj_shapes[i].mesh.material_ids[f] < materials.size())
				{
					triangle = Triangle(pos[0], pos[1], pos[2], nor[0], nor[1], nor[2], t0_, t1_, t2_, materials[obj_shapes[i].mesh.material_ids[f]]);
				}
				else
				{
					triangle = Triangle(pos[0], pos[1], pos[2], nor[0], nor[1], nor[2], t0_, t1_, t2_, material);
				}
				triangles->push_back(triangle);

				index_offset += fnum;
			}
		}
		this->count = triangles->size();
		this->triangles = triangles->data();
	}
	__host__  Mesh(vec3 position, Triangle* triangles = nullptr, int count = 0, Material material = Material())
	{
		this->position = position;
		this->triangles = new Triangle[count];
		this->count = count;
		for (int i = 0; i < count; i++)
		{
			this->triangles[i] = triangles[i];
			this->triangles[i].material = material;
		}
	}
	vec3 position;
	Triangle* triangles;
	int count;
	KDTree* tree;
	KDTreeNode* nodes;	
	int* tna;
	__device__ ObjectIntersection Intersect(Ray ray)
	{
#if ENABLE_KDTREE
		ObjectIntersection intersection = ObjectIntersection();
		intersection = RayKDTreeTraversal(nodes, tna, ray, triangles, position);
		if (intersection.hit == true)
			intersection.hitPtr = this;
		return intersection;
#else
		float tNear = INFINITY;
		ObjectIntersection intersection = ObjectIntersection();
		for (int i = 0; i < count; i++)
		{
			ObjectIntersection temp = triangles[i].Intersect(ray, position);
			if (temp.hit && temp.t < tNear)
			{
				tNear = temp.t;
				intersection = temp;
			}
		}
		if (intersection.hit == true)
			intersection.hitPtr = this;
		return intersection;
#endif
	}
};

#pragma endregion Structs

#pragma region Scene

Mesh CreateBox(vec3 pos, vec3 halfExtents, Material material)
{
	float halfWidth = halfExtents[0];
	float halfHeight = halfExtents[1];
	float halfDepth = halfExtents[2];

	vec3 vertices[8] =
	{
		vec3(halfWidth, halfHeight, halfDepth),
		vec3(-halfWidth, halfHeight, halfDepth),
		vec3(halfWidth, -halfHeight, halfDepth),
		vec3(-halfWidth, -halfHeight, halfDepth),
		vec3(halfWidth, halfHeight, -halfDepth),
		vec3(-halfWidth, halfHeight, -halfDepth),
		vec3(halfWidth, -halfHeight, -halfDepth),
		vec3(-halfWidth, -halfHeight, -halfDepth)
	};

	static int indices[36] =
	{
		0, 1, 2, 3, 2, 1, 4, 0, 6,
		6, 0, 2, 5, 1, 4, 4, 1, 0,
		7, 3, 1, 7, 1, 5, 5, 4, 7,
		7, 4, 6, 7, 2, 3, 7, 6, 2
	};

	std::vector<Triangle> triangles;

	for (int i = 0; i < 36; i += 3)
	{
		//triangles.push_back(Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]], material));
	}

	return Mesh(pos, triangles.data(), 12, material);
}

dim3 block, grid;
Camera* camera;

Sphere spheres[] =
{
	//Sphere(vec3(20, 10, 14), 8, Material(TRANS,  vec3(1))),
	//Sphere(vec3(-14, 8, -20), 8, Material(DIFF,  vec3(1))),
	//Sphere(vec3(-14, 8, 14), 8, Material(SPEC,  vec3(1))),
	//Sphere(vec3(14, 8, -14), 8, Material(GLOSS,  vec3(1)))
	//Sphere(vec3(0, 65, 0), 8, Material(DIFF, vec3(0.75, 0.75, 0.75), vec3(2.2, 2.2, 2.2))),
	//Sphere(vec3(0, 30, 0), 8,  Material(TRANS,  vec3(1)))

	//Sphere(vec3(-30, 8, 0), 8, Material(TRANS,  vec3(1))),
	Sphere(vec3(0, 45, 0), 1.0f, Material(TRANS,  vec3(1), vec3(2.2f, 2.2f, 2.2f))),
	Sphere(vec3(-10, 8, -10), 8, Material(SPEC,  vec3(1))),
	Sphere(vec3(-10, 8, 10), 8, Material(TRANS,  vec3(1)))
};
Mesh meshes[] =
{
	Mesh(vec3(0,0,0), "Cornell.obj", Material(DIFF))
	//Mesh(vec3(0,0,0), "Cornell_Water0.obj", Material(DIFF)),
	//Mesh(vec3(0,0,0), "Cornell_Water1.obj", Material(SPEC)),
	//Mesh(vec3(0,0,0), "Cornell_Water2.obj", Material(TRANS))
	//Mesh(vec3(0,0,0), "Sponza.obj")
	//Mesh(vec3(0,0,0), "test.obj", Material(TRANS, vec3(1)))
	//Mesh(vec3(150,-50,150), "BigDragon.obj", Material(DIFF, vec3(1)))
	//Mesh(vec3(0,0,0), "Cornell_Small.obj")
	//Mesh(vec3(0,5,0), "wired_mesh.obj", Material(TRANS, vec3(1)))
	//CreateBox(vec3(0, 30, 0), vec3(30, 1, 30), Material(DIFF, vec3(0.75, 0.75, 0.75), vec3(2.2, 2.2, 2.2))),
	//Mesh(vec3(0, 0, 0), "board.obj", Material(DIFF))
	//Mesh(vec3(0, 3, 0), "Crystal_Low.obj", Material(TRANS)),
	////CreateBox(vec3(0, 0, 0), vec3(30, 1, 30), Material(DIFF, vec3(0.75, 0.75, 0.75))),
	//CreateBox(vec3(30, 15, 0), vec3(1, 15, 30), Material(DIFF, vec3(0.0, 0.0, 0.75))),
	//CreateBox(vec3(-30, 15, 0), vec3(1, 15, 30), Material(DIFF, vec3(0.75, 0.0, 0.0))),
	//CreateBox(vec3(0, 15, 30), vec3(30, 15, 1), Material(DIFF, vec3(0.75, 0.75, 0.75))),
	//CreateBox(vec3(0, 15, -30), vec3(30, 15, 1), Material(DIFF, vec3(0.75, 0.75, 0.75)))
};

#pragma endregion Scene

#pragma region Kernels

__device__ ObjectIntersection Intersect(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp = ObjectIntersection();

	for (int i = 0; i < sphereCount; i++)
	{
		temp = spheres[i].Intersect(ray);

		if (temp.hit)
		{
			if (intersection.t == 0 || temp.t < intersection.t)
			{
				intersection = temp;
			}
		}
	}

	for (int i = 0; i < meshCount; i++)
	{
		temp = meshes[i].Intersect(ray);

		if (temp.hit)
		{
			if (intersection.t == 0 || temp.t < intersection.t)
			{
				intersection = temp;
			}
		}
	}

	return intersection;
}

__device__ Ray GetReflectedRay(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, vec3 hitPoint, glm::vec3 normal, vec3 &mask, Material material, curandState* randState)
{
	switch (material.type)
	{
	case DIFF:
	{
		vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
		float r1 = two_pi<float>() * curand_uniform(randState);
		float r2 = curand_uniform(randState);
		float r2s = sqrt(r2);

		vec3 w = nl;
		vec3 u;
		if (fabs(w.x) > 0.1f)
			u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
		else
			u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
		vec3 v = cross(w, u);
		vec3 reflected = normalize((u * __cosf(r1) * r2s + v * __sinf(r1) * r2s + w * sqrt(1 - r2)));
		mask *= material.color;
		return Ray(hitPoint, reflected);
	}
	case GLOSS:
	{
		float phi = 2 * pi<float>() * curand_uniform(randState);
		float r2 = curand_uniform(randState);
		float phongExponent = 20;
		float cosTheta = __powf(1 - r2, 1.0f / (phongExponent + 1));
		float sinTheta = __sinf(1 - cosTheta * cosTheta);

		vec3 w = normalize(ray.direction - normal * 2.0f * dot(normal, ray.direction));
		vec3 u = normalize(cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
		vec3 v = cross(w, u);

		vec3 reflected = normalize(u * __cosf(phi) * sinTheta + v * __sinf(phi) * sinTheta + w * cosTheta);
		mask *= material.color;
		return Ray(hitPoint, reflected);
	}
	case TRANS:
	{
		vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
		vec3 reflection = ray.direction - normal * 2.0f * dot(normal, ray.direction);
		bool into = dot(normal, nl) > EPSILON;
		float nc = 1.0f;
		float nt = 1.5f;
		float nnt = into ? nc / nt : nt / nc;

		float Re, RP, TP, Tr;
		vec3 tdir = vec3(0.0f, 0.0f, 0.0f);

		float ddn = dot(ray.direction, nl);
		float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

		if (cos2t < EPSILON) return Ray(hitPoint, reflection);

		if (into)
			tdir = normalize((ray.direction * nnt - normal * (ddn * nnt + sqrt(cos2t))));
		else
			tdir = normalize((ray.direction * nnt + normal * (ddn * nnt + sqrt(cos2t))));

		float a = nt - nc;
		float b = nt + nc;
		float R0 = a * a / (b * b);

		float c;
		if (into)
			c = 1 + ddn;
		else
			c = 1 - dot(tdir, normal);

		Re = R0 + (1 - R0) * c * c * c * c * c;
		Tr = 1 - Re;

		float P = .25 + .5 * Re;
		RP = Re / P;
		TP = Tr / (1 - P);

		if (curand_uniform(randState) < P)
		{
			mask *= (RP);
			return Ray(hitPoint, reflection);
		}
		mask *= (TP);
		return Ray(hitPoint, tdir);
	}
	case SPEC:
	{
		vec3 reflected = ray.direction - normal * 2.0f * dot(normal, ray.direction);
		mask *= material.color;
		return Ray(hitPoint, reflected);
	}
	}
}

// Photon Map (Building Photon Map)
__device__ Photon TraceRay(Ray ray, vec3 lightEmission, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, curandState* randState)
{
	/*vec3 resultColor = lightEmission;
	MaterialType beforeType = NONE;
	for (int depth = 0; depth < 4; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);

		if (intersection.hit == false) return Photon();

		vec3 color = intersection.material.color;
		float maxReflection = color.x > color.y && color.x > color.z ? color.x : color.y > color.z ? color.y : color.z;
		float random = curand_uniform(randState);

		vec3 position = ray.origin + ray.direction * intersection.t;

		if (intersection.material.type == DIFF || intersection.material.type == GLOSS || intersection.material.type == SPEC)
		{
			if (beforeType == TRANS)
			{
				Photon photon = Photon();
				photon.isHit = intersection.hit;
				photon.normal = intersection.normal;
				photon.position = position;
				photon.type = intersection.material.type;
				photon.power = resultColor;
				return photon;
			}
		}
		beforeType = intersection.material.type;
		ray = GetReflectedRay(ray, spheres, meshes, sphereCount, meshCount, position, intersection.normal, resultColor, intersection.material, randState);
	}*/
	return Photon();
}

// Path Tracing + Photon Map
__device__ vec3 TraceRay(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, bool directLighting, float directLightingConstant, Photon* map, int maxPhotons, curandState* randState)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < MAX_DEPTH; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);

		if (intersection.hit == 0)
		{
			float longlatX = atan2(ray.direction.x, ray.direction.z);
			longlatX = longlatX < EPSILON ? longlatX + two_pi<float>() : longlatX;
			float longlatY = acos(-ray.direction.y);

			float u = longlatX / two_pi<float>();
			float v = longlatY / pi<float>();

			int u2 = (int)(u * HDRWidth);
			int tvec = (int)(v * HDRHeight);

			int HDRtexelidx = u2 + tvec * HDRWidth;

			float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);
			vec3 HDRcol2 = vec3(HDRcol.x, HDRcol.y, HDRcol.z);

			return resultColor + (mask * HDRcol2);
		}

		vec3 hitPoint = ray.origin + ray.direction * intersection.t;
		vec3 emission = intersection.material.emission;
		vec3 photonColor = vec3(0, 0, 0);
		int nearPhotonCount = 0;
		for (int i = 0; i < maxPhotons; i++)
		{
			float dist = distance(hitPoint, map[i].position);
			if (dist <= 5.0f && dot(intersection.normal, map[i].normal) > 0)
			{
				photonColor += map[i].power * 1.0f / (pi<float>() * dist);
				nearPhotonCount++;
			}
		}
		if (nearPhotonCount >= 3)
			photonColor /= (float)nearPhotonCount;

		if (intersection.material.type == DIFF || intersection.material.type == GLOSS || intersection.material.type == SPEC)
			emission += photonColor;

		vec3 explicitLightColor = vec3(0, 0, 0);

		if (directLighting)
		{
			if (intersection.material.type == DIFF)
			{
				for (int i = 0; i < sphereCount; i++)
				{
					float maxEmission = max(max(spheres[i].material.emission.x, spheres[i].material.emission.y), spheres[i].material.emission.z);
					if (maxEmission < 1.0f)
						continue;

					vec3 lightPoint = spheres[i].RandomPoint(randState);
					vec3 lightDirection = normalize(lightPoint - hitPoint);
					Ray rayToLight = Ray(hitPoint, lightDirection);
					ObjectIntersection lightIntersection = Intersect(rayToLight, spheres, meshes, sphereCount, meshCount);
					if (lightIntersection.hitPtr == &spheres[i])
					{
						float wi = dot(lightDirection, intersection.normal);
						if (wi > 0)
						{
							float radius = spheres[i].radius;
							float cosMax = sqrt(1 - radius * radius / dot(hitPoint - lightPoint, hitPoint - lightPoint));
							float omega = two_pi<float>() * (1 - cosMax);
							explicitLightColor += spheres[i].material.emission * wi * omega * one_over_pi<float>() * directLightingConstant;
						}
					}
				}
			}
		}

		resultColor += mask * (emission + explicitLightColor);
		ray = GetReflectedRay(ray, spheres, meshes, sphereCount, meshCount, hitPoint, intersection.normal, mask, intersection.material, randState);
	}
	return resultColor;
}

// Real time + Photon Mapping Kernel
__global__ void PathKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int loopX, int loopY, bool dof, bool directLighting, float directLightingConstant, int frame, Photon* map, int mapSize, cudaSurfaceObject_t surface)
{
	int width = camera->width;
	int height = camera->height;
	int x = gridDim.x * blockDim.x * loopX + blockIdx.x * blockDim.x + threadIdx.x;
	int y = gridDim.y * blockDim.y * loopY + blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int i = y * width + x;

	if (i >= width * height) return;

	curandState randState;
	float4 originColor;
	surf2Dread(&originColor, surface, x * sizeof(float4), y);

	vec3 resultColor = vec3(0, 0, 0);
	curand_init(WangHash(threadId) + WangHash(frame), 0, 0, &randState);
	Ray ray = camera->GetRay(&randState, x, y, dof);
	vec3 color = TraceRay(ray, spheres, meshes, sphereCount, meshCount, directLighting, directLightingConstant, map, mapSize, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float)(frame - 1) + color) / (float)frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

// Kernel to Build Photon Map
__global__ void PhotonMapKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, vec3 lightPos, vec3 lightEmission, int maxPhotons, Photon* map, int frame = 0)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i >= maxPhotons) return;

	curandState randState;

	Photon photon;
	int threshold = MAX_BUILD_PHOTON_TRHESHOLD;
	int count = 0;
	while (!photon.isHit)
	{
		if (count > threshold)
			break;
		curand_init(WangHash(WangHash(i) + WangHash(count) + WangHash(maxPhotons) + WangHash(frame)), 0, 0, &randState);
		count++;

		float theta = 2 * pi<float>() * curand_uniform(&randState);
		float phi = acos(1 - 2 * curand_uniform(&randState));
		vec3 dir = normalize(vec3(__sinf(phi) * __cosf(theta), __sinf(phi) * __sinf(theta), __cosf(phi)));
		Ray ray = Ray(lightPos, dir);
		photon = TraceRay(ray, lightEmission, spheres, meshes, sphereCount, meshCount, &randState);
	}
	map[i] = photon;
}

// Photon Mapping Rendering Loop
void TracingLoop(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int frame, bool dof, bool directLighting, float directLightingConstant, Photon* map, int mapSize, cudaSurfaceObject_t surface)
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 5000000000 * sizeof(float));
	tracingGridProgress = 0;
	for (int i = 0; i < TRACE_OUTER_LOOP_X; i++)
	{
		for (int j = 0; j < TRACE_OUTER_LOOP_Y; j++)
		{
			PathKernel << <grid, block >> > (camera, spheres, meshes, sphereCount, meshCount, i, j, dof, directLighting, directLightingConstant, frame, map, mapSize, surface);
			gpuErrorCheck(cudaDeviceSynchronize());
		}
	}
}

Photon* BuildPhotonMap(int maxPhotons, int frame = 0)
{
	Photon *photons, *cudaPhotonMap/*, *cudaDebugPhotonMap*/;

	vec3 lightPos = vec3(0, 50, 0);
	vec3 lightEmission = vec3(1.5, 1.5, 1.5);

	block = dim3(256, 1);
	grid.x = ceil(maxPhotons / block.x);
	grid.y = 1;

	gpuErrorCheck(cudaMalloc(&cudaPhotonMap, sizeof(Photon) * maxPhotons));
	//gpuErrorCheck(cudaMalloc(&cudaDebugPhotonMap, sizeof(Photon) * maxPhotons));

	cudaEvent_t start, stop;
	float memoryAllocTime, renderingTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	Camera* cudaCamera;
	gpuErrorCheck(cudaMalloc(&cudaCamera, sizeof(Camera)));
	gpuErrorCheck(cudaMemcpy(cudaCamera, camera, sizeof(Camera), cudaMemcpyHostToDevice));

	int sphereCount = sizeof(spheres) / sizeof(Sphere);
	Sphere* cudaSpheres;
	gpuErrorCheck(cudaMalloc(&cudaSpheres, sizeof(Sphere) * sphereCount));
	gpuErrorCheck(cudaMemcpy(cudaSpheres, spheres, sizeof(Sphere) * sphereCount, cudaMemcpyHostToDevice));

	int meshCount = sizeof(meshes) / sizeof(Mesh);
	Mesh* cudaMeshes;
	std::vector<Mesh>* meshVector = new std::vector<Mesh>;
	std::vector<Triangle*> triangleVector;
	for (int i = 0; i < meshCount; i++)
	{
		Mesh currentMesh = meshes[i];
		Mesh cudaMesh = currentMesh;
		Triangle* cudaTriangles;
		gpuErrorCheck(cudaMalloc(&cudaTriangles, sizeof(Triangle) * currentMesh.count));
		gpuErrorCheck(cudaMemcpy(cudaTriangles, currentMesh.triangles, sizeof(Triangle) * currentMesh.count, cudaMemcpyHostToDevice));
		cudaMesh.triangles = cudaTriangles;
		meshVector->push_back(cudaMesh);
		triangleVector.push_back(cudaTriangles);
	}
	gpuErrorCheck(cudaMalloc(&cudaMeshes, sizeof(Mesh) * meshCount));
	gpuErrorCheck(cudaMemcpy(cudaMeshes, meshVector->data(), sizeof(Mesh) * meshCount, cudaMemcpyHostToDevice));

	gpuErrorCheck(cudaDeviceSynchronize());
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memoryAllocTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	PhotonMapKernel << <grid, block >> > (cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, lightPos, lightEmission, maxPhotons, cudaPhotonMap, frame);
	gpuErrorCheck(cudaDeviceSynchronize());
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&renderingTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Building Photon Map End | Memory Allocation Time : %f ms | Building time : %f ms\n", memoryAllocTime, renderingTime);

	photons = new Photon[maxPhotons];
	gpuErrorCheck(cudaMemcpy(photons, cudaPhotonMap, sizeof(Photon) * maxPhotons, cudaMemcpyDeviceToHost));

	cudaFree(cudaPhotonMap);
	cudaFree(cudaCamera);
	cudaFree(cudaSpheres);
	for (auto & triangle : triangleVector)
	{
		cudaFree(triangle);
	}
	for (auto & mesh : *meshVector)
	{
		cudaFree(&mesh);
	}
	delete meshVector;
	cudaFree(cudaMeshes);
	return photons;
}

void RenderRealTime(cudaSurfaceObject_t surface, bool dof, bool photon, bool directLighting, int frame)
{
	int width = camera->width;
	int height = camera->height;

	cudaEvent_t start, stop;
	gpuErrorCheck(cudaEventCreate(&start));
	gpuErrorCheck(cudaEventRecord(start, 0));

	Camera* cudaCamera;
	gpuErrorCheck(cudaMalloc(&cudaCamera, sizeof(Camera)));
	gpuErrorCheck(cudaMemcpy(cudaCamera, camera, sizeof(Camera), cudaMemcpyHostToDevice));

	int sphereCount = sizeof(spheres) / sizeof(Sphere);
	Sphere* cudaSpheres;
	gpuErrorCheck(cudaMalloc(&cudaSpheres, sizeof(Sphere) * sphereCount));
	gpuErrorCheck(cudaMemcpy(cudaSpheres, spheres, sizeof(Sphere) * sphereCount, cudaMemcpyHostToDevice));

	int meshCount = sizeof(meshes) / sizeof(Mesh);
	Mesh* cudaMeshes;
	std::vector<Mesh> meshVector;
	std::vector<Triangle*> triangleVector;
	for (int i = 0; i < meshCount; i++)
	{
		Mesh currentMesh = meshes[i];
		Mesh cudaMesh = currentMesh;
		Triangle* cudaTriangles;
		gpuErrorCheck(cudaMalloc(&cudaTriangles, sizeof(Triangle) * currentMesh.count));
		gpuErrorCheck(cudaMemcpy(cudaTriangles, currentMesh.triangles, sizeof(Triangle) * currentMesh.count, cudaMemcpyHostToDevice));

#if ENABLE_KDTREE
		cudaMesh.nodes = currentMesh.tree->nodes.data;
		cudaMesh.tna = currentMesh.tree->triangleNodeAssociation.data;
#endif
		cudaMesh.triangles = cudaTriangles;
		meshVector.push_back(cudaMesh);
		triangleVector.push_back(cudaTriangles);
	}

	gpuErrorCheck(cudaMalloc(&cudaMeshes, sizeof(Mesh) * meshCount));
	gpuErrorCheck(cudaMemcpy(cudaMeshes, meshVector.data(), sizeof(Mesh) * meshCount, cudaMemcpyHostToDevice));

	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaEventCreate(&stop));
	gpuErrorCheck(cudaEventRecord(stop, 0));
	gpuErrorCheck(cudaEventSynchronize(stop));
	gpuErrorCheck(cudaEventElapsedTime(&memoryAllocTime, start, stop));
	gpuErrorCheck(cudaEventDestroy(start));
	gpuErrorCheck(cudaEventDestroy(stop));

	int photonMapSize = 0;
	Photon* cudaPhotonMap;
	Photon* photonMap;
	if (photon)
	{
		photonMapSize = MAX_PHOTONS;
		photonMap = BuildPhotonMap(photonMapSize, frame);

		gpuErrorCheck(cudaMalloc(&cudaPhotonMap, sizeof(Photon) * photonMapSize));
		gpuErrorCheck(cudaMemcpy(cudaPhotonMap, photonMap, sizeof(Photon) * photonMapSize, cudaMemcpyHostToDevice));
	}

	block = dim3(16, 9);
	grid.x = ceil(ceil(width / TRACE_OUTER_LOOP_X) / block.x);
	grid.y = ceil(ceil(height / TRACE_OUTER_LOOP_Y) / block.y);

	gpuErrorCheck(cudaEventCreate(&start));
	gpuErrorCheck(cudaEventRecord(start, 0));
	TracingLoop(cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, frame, dof, directLighting, directLightingConstant, cudaPhotonMap, photonMapSize, surface);
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaEventCreate(&stop));
	gpuErrorCheck(cudaEventRecord(stop, 0));
	gpuErrorCheck(cudaEventSynchronize(stop));
	gpuErrorCheck(cudaEventElapsedTime(&renderingTime, start, stop));
	gpuErrorCheck(cudaEventDestroy(start));
	gpuErrorCheck(cudaEventDestroy(stop));

	if (photon)
	{
		delete photonMap;
		cudaFree(cudaPhotonMap);
	}
	gpuErrorCheck(cudaFree(cudaCamera));
	gpuErrorCheck(cudaFree(cudaSpheres));
	for (auto & triangle : triangleVector)
	{
		gpuErrorCheck(cudaFree(triangle));
	}
	gpuErrorCheck(cudaFree(cudaMeshes));
}

#pragma endregion Kernels

#pragma region Opengl Callbacks

	void Keyboard(unsigned char key, int x, int y)
	{
		keyState[key] = true;
		mousePos[0] = x;
		mousePos[1] = y;

		if (IsKeyDown('r'))
		{
			enableDof = !enableDof;
			cudaDirty = true;
		}
		if (IsKeyDown('b'))
		{
			enablePhoton = !enablePhoton;
			cudaDirty = true;
		}
		if (IsKeyDown('q'))
		{
			enableSaveImage = true;
			frame = 1;
			cudaDirty = false;
			cudaToggle = true;
		}
		if (IsKeyDown('f'))
		{
			cudaToggle = !cudaToggle;
			frame = 1;
			cudaDirty = false;
		}
		if (IsKeyDown('n'))
		{
			enableDrawNormal = !enableDrawNormal;
			cudaToggle = false;
		}
		if (IsKeyDown('k'))
		{
			enableDrawKDTree = !enableDrawKDTree;
			cudaToggle = false;
		}
		if (IsKeyDown('l'))
		{
			enableDirectLighting = !enableDirectLighting;
			cudaDirty = true;
		}
		if (IsKeyDown('t'))
		{
			camera->aperture += 0.1f;
			cudaDirty = true;
			printf("%f %f\n", camera->aperture, camera->focalDistance);
		}
		if (IsKeyDown('g'))
		{
			camera->aperture -= 0.1f;
			cudaDirty = true;
			printf("%f %f\n", camera->aperture, camera->focalDistance);
		}
		if (IsKeyDown('y'))
		{
			camera->focalDistance += 0.5f;
			cudaDirty = true;
			printf("%f %f\n", camera->aperture, camera->focalDistance);
		}
		if (IsKeyDown('h'))
		{
			camera->focalDistance -= 0.5f;
			cudaDirty = true;
			printf("%f %f\n", camera->aperture, camera->focalDistance);
		}
		if (IsKeyDown('p'))
		{
			printf("Camera Position : %f %f %f\n", camera->position.x, camera->position.y, camera->position.z);
			printf("Pitch Yaw : %f %f\n", camera->pitch, camera->yaw);
		}
		if (IsKeyDown('u'))
		{
			enableGUI = !enableGUI;
		}
		ImGuiIO& io = ImGui::GetIO();
		io.AddInputCharacter(key);
		glutPostRedisplay();
	}
	void KeyboardUp(unsigned char key, int x, int y)
	{
		keyState[key] = false;
		mousePos[0] = x;
		mousePos[1] = y;
		glutPostRedisplay();
	}
	void Special(int key, int x, int y)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.AddInputCharacter(key);
		glutPostRedisplay();
	}
	void SpecialUp(int key, int x, int y)
	{
		glutPostRedisplay();
	}
	void Mouse(int button, int state, int x, int y)
	{
		mousePos[0] = x;
		mousePos[1] = y;
		mouseState[button] = !state;

		ImGuiIO& io = ImGui::GetIO();
		io.MousePos = ImVec2(float(x), float(y));

		if (state == GLUT_DOWN && (button == GLUT_LEFT_BUTTON))
			io.MouseDown[0] = true;
		else
			io.MouseDown[0] = false;

		if (state == GLUT_DOWN && (button == GLUT_RIGHT_BUTTON))
			io.MouseDown[1] = true;
		else
			io.MouseDown[1] = false;

		glutPostRedisplay();
	}
	void MouseWheel(int button, int dir, int x, int y)
	{
		if (dir > 0)
		{
			camera->fov++;
			cudaDirty = true;
		}
		else
		{
			camera->fov--;
			cudaDirty = true;
		}
		glutPostRedisplay();
	}
	void Motion(int x, int y)
	{
		mousePos[0] = x;
		mousePos[1] = y;
		ImGuiIO& io = ImGui::GetIO();
		io.MousePos = ImVec2(float(x), float(y));
		glutPostRedisplay();
	}
	void Reshape(int w, int h)
	{
		camera->UpdateScreen(w, h);
	}
	void Idle()
	{
		int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
		deltaTime = (timeSinceStart - oldTimeSinceStart) * 0.001f;
		oldTimeSinceStart = timeSinceStart;
		glutPostRedisplay();
	}
	void Display(void)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		camera->UpdateCamera(deltaTime);

		// OpenGL Draw
		if (cudaToggle)
		{
			int width = camera->width;
			int height = camera->height;
			glColor3f(1, 1, 1);
			glDisable(GL_LIGHTING);
			cudaResourceDesc viewCudaArrayResourceDesc;
			{
				viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
				viewCudaArrayResourceDesc.res.array.array = viewArray;
			}

			cudaSurfaceObject_t viewCudaSurfaceObject;
			cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
			{
				if (cudaDirty)
				{
					frame = 0;
					cudaDirty = false;
				}
				RenderRealTime(viewCudaSurfaceObject, enableDof, enablePhoton, enableDirectLighting, ++frame);
			}
			cudaDestroySurfaceObject(viewCudaSurfaceObject);

			cudaGraphicsUnmapResources(1, &viewResource);

			cudaStreamSynchronize(0);

			glLoadIdentity();
			glViewport(0, 0, width, height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, width, 0, height, -1000, 1000);

			glBindTexture(GL_TEXTURE_2D, viewGLTexture);
			{
				glBegin(GL_QUADS);
				{
					glTexCoord2f(0, 1);	glVertex2f(0, 0);
					glTexCoord2f(1, 1);	glVertex2f(width, 0);
					glTexCoord2f(1, 0);	glVertex2f(width, height);
					glTexCoord2f(0, 0);	glVertex2f(0, height);
				}
				glEnd();
			}

			if (enableSaveImage && frame >= imageSaveSamples)
			{
				enableSaveImage = false;
				cudaToggle = false;
				cudaDirty = false;
				isSavingImage = false;
				frame = 1;

				GLubyte *pixels = new GLubyte[3 * width*height];
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
				FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);
				SwapRedBlue32(image);
				FreeImage_Save(FIF_PNG, image, "Result.png", 0);
				FreeImage_Unload(image);
				delete pixels;
			}

			glBindTexture(GL_TEXTURE_2D, 0);
			glFinish();
			glEnable(GL_LIGHTING);
		}
		else
		{
			// Draw Opengl
			{
				int size = sizeof(spheres) / sizeof(Sphere);
				for (int n = 0; n < size; n++)
				{
					glPushMatrix();
					glTranslatef(spheres[n].position.x, spheres[n].position.y, spheres[n].position.z);
					glColor3fv(value_ptr(spheres[n].material.color));
					int i, j;
					int lats = 50;
					int longs = 50;
					float radius = spheres[n].radius;
					for (i = 0; i <= lats; i++)
					{
						float lat0 = pi<float>() * (-float(0.5) + (float) (i - 1) / lats);
						float z0 = radius * sin(lat0);
						float zr0 = radius * cos(lat0);

						float lat1 = pi<float>() * (-float(0.5) + (float) i / lats);
						float z1 = radius * sin(lat1);
						float zr1 = radius * cos(lat1);

						glBegin(GL_QUAD_STRIP);
						for (j = 0; j <= longs; j++)
						{
							float lng = 2 * pi<float>() * (float) (j - 1) / longs;
							float x = cos(lng);
							float y = sin(lng);
							glNormal3f(x * zr1, y * zr1, z1);
							glVertex3f(x * zr1, y * zr1, z1);
							glNormal3f(x * zr0, y * zr0, z0);
							glVertex3f(x * zr0, y * zr0, z0);
						}
						glEnd();
					}
					glPopMatrix();
				}
				size = sizeof(meshes) / sizeof(Mesh);
				for (int n = 0; n < size; n++)
				{
					glPushMatrix();
					glTranslatef(meshes[n].position.x, meshes[n].position.y, meshes[n].position.z);
					Triangle* triangles = meshes[n].triangles;
					for (int i = 0; i < meshes[n].count; i++)
					{
						glColor3fv(value_ptr(triangles[i].material.color));
						vec3 p0 = triangles[i].pos[0];
						vec3 p1 = triangles[i].pos[1];
						vec3 p2 = triangles[i].pos[2];

						vec3 normal = cross((p2 - p0), (p1 - p0));
						normal = normalize(normal);
						glBegin(GL_TRIANGLE_STRIP);
						glNormal3fv(value_ptr(normal));
						glVertex3fv(value_ptr(p0));
						glVertex3fv(value_ptr(p1));
						glVertex3fv(value_ptr(p2));
						glEnd();

						if (enableDrawNormal)
						{
							glLineWidth(1.0f);
							glColor3f(1.0f, 1.0f, 1.0f);
							glBegin(GL_LINES);
							glVertex3fv(value_ptr(triangles[i].pos[0]));
							glVertex3fv(value_ptr(triangles[i].nor[0] + triangles[i].pos[0]));
							glVertex3fv(value_ptr(triangles[i].pos[1]));
							glVertex3fv(value_ptr(triangles[i].nor[1] + triangles[i].pos[1]));
							glVertex3fv(value_ptr(triangles[i].pos[2]));
							glVertex3fv(value_ptr(triangles[i].nor[2] + triangles[i].pos[2]));
							glEnd();
						}
					}
					if (enableDrawKDTree)
					{
						glDisable(GL_LIGHTING);
						int nodeSize = meshes[n].tree->nodes.size();
						glLineWidth(1.0f);
						KDTreeNode* nodes = new KDTreeNode[nodeSize];
						meshes[n].tree->nodes.CopyToHost(nodes);
						for (int i = 0; i < meshes[n].tree->nodes.size(); i++)
						{
							if (nodes[i].depth > KDTREE_MAX_DEPTH)
								printf("WHAT %d\n", nodes[i].depth);
							AABB box = nodes[i].nodeAABB;

							vec3 corner[8];

							corner[0] = { box.bounds[0].x, box.bounds[0].y, box.bounds[0].z };
							corner[1] = { box.bounds[1].x, box.bounds[0].y, box.bounds[0].z };
							corner[2] = { box.bounds[1].x, box.bounds[0].y, box.bounds[1].z };
							corner[3] = { box.bounds[0].x, box.bounds[0].y, box.bounds[1].z };
							corner[4] = { box.bounds[0].x, box.bounds[1].y, box.bounds[0].z };
							corner[5] = { box.bounds[1].x, box.bounds[1].y, box.bounds[0].z };
							corner[6] = { box.bounds[1].x, box.bounds[1].y, box.bounds[1].z };
							corner[7] = { box.bounds[0].x, box.bounds[1].y, box.bounds[1].z };

							glColor3f(1.0f, 1 - (i / float(nodeSize)), 0.0f);
							glLineWidth(i / float(nodeSize));

							glBegin(GL_LINES);

							glVertex3f(corner[0].x, corner[0].y, corner[0].z);
							glVertex3f(corner[1].x, corner[1].y, corner[1].z);

							glVertex3f(corner[1].x, corner[1].y, corner[1].z);
							glVertex3f(corner[2].x, corner[2].y, corner[2].z);

							glVertex3f(corner[2].x, corner[2].y, corner[2].z);
							glVertex3f(corner[3].x, corner[3].y, corner[3].z);

							glVertex3f(corner[3].x, corner[3].y, corner[3].z);
							glVertex3f(corner[0].x, corner[0].y, corner[0].z);



							glVertex3f(corner[0].x, corner[0].y, corner[0].z);
							glVertex3f(corner[4].x, corner[4].y, corner[4].z);

							glVertex3f(corner[1].x, corner[1].y, corner[1].z);
							glVertex3f(corner[5].x, corner[5].y, corner[5].z);

							glVertex3f(corner[2].x, corner[2].y, corner[2].z);
							glVertex3f(corner[6].x, corner[6].y, corner[6].z);

							glVertex3f(corner[3].x, corner[3].y, corner[3].z);
							glVertex3f(corner[7].x, corner[7].y, corner[7].z);



							glVertex3f(corner[4].x, corner[4].y, corner[4].z);
							glVertex3f(corner[5].x, corner[5].y, corner[5].z);

							glVertex3f(corner[5].x, corner[5].y, corner[5].z);
							glVertex3f(corner[6].x, corner[6].y, corner[6].z);

							glVertex3f(corner[6].x, corner[6].y, corner[6].z);
							glVertex3f(corner[7].x, corner[7].y, corner[7].z);

							glVertex3f(corner[7].x, corner[7].y, corner[7].z);
							glVertex3f(corner[4].x, corner[4].y, corner[4].z);

							glEnd();
						}
						delete[] nodes;
						glEnable(GL_LIGHTING);
					}
					glPopMatrix();
				}
			}

		}


		int width = camera->width;
		int height = camera->height;
		ImGui_ImplGLUT_NewFrame(width, height);

		// UI
		if (enableGUI)
		{
			ImGui::Begin("Cuda Tracer", nullptr, ImVec2(0,0), -1.0f, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_Once);
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			if (cudaToggle)
			{
				ImGui::Text("Current Frame : %d", frame);

				if (isSavingImage)
				{
					ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
					ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
				}

				if (ImGui::Button("Save Image"))
				{
					GLubyte *pixels = new GLubyte[3 * width*height];
					glPixelStorei(GL_PACK_ALIGNMENT, 1);
					glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
					FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);
					SwapRedBlue32(image);
					stringstream ss;
					ss << "Result_" << frame << ".png";
					FreeImage_Save(FIF_PNG, image, ss.str().c_str(), 0);
					FreeImage_Unload(image);
					delete pixels;
				}
				ImGui::Text("Memory Allocation Time : %f ms", memoryAllocTime);
				ImGui::Text("Rendering time : %f ms", renderingTime);
				ImGui::Text("Total Time : %f ms", memoryAllocTime + renderingTime);

				if (ImGui::Checkbox("Enable Dof", &enableDof))
					cudaDirty = true;
				if (enableDof)
				{
					if (ImGui::SliderFloat("Focal Distance", &(camera->focalDistance), EPSILON, 500))
						cudaDirty = true;
					if (ImGui::SliderFloat("Aperture", &(camera->aperture), EPSILON, 50))
						cudaDirty = true;
				}
				if (ImGui::Checkbox("Enable Direct Lighting", &enableDirectLighting))
					cudaDirty = true;
				if (ImGui::Checkbox("Enable Photon Mapping", &enablePhoton))
					cudaDirty = true;
				if (ImGui::SliderFloat("Direct Lighting Weight", &directLightingConstant, EPSILON, 1000.0f))
					cudaDirty = true;

				if (isSavingImage)
				{
					ImGui::PopItemFlag();
					ImGui::PopStyleVar();
				}
			}
			else
			{
				ImGui::InputInt("Image Samples", &imageSaveSamples, 1, 100);
				ImGui::SameLine();
				if (ImGui::Button("Save Image"))
				{
					enableSaveImage = true;
					frame = 1;
					cudaDirty = false;
					cudaToggle = true;
					isSavingImage = true;
				}

				if (ImGui::Checkbox("Draw Normal", &enableDrawNormal))
					cudaDirty = true;
				if (ImGui::Checkbox("Draw Debug KDTree AABBox", &enableDrawKDTree))
					cudaDirty = true;
			}

			if (!isSavingImage)
			{
				int sphereCount = sizeof(spheres) / sizeof(Sphere);
				int meshCount = sizeof(meshes) / sizeof(Mesh);

				if (ImGui::CollapsingHeader("Objects"))
				{
					ImGui::Text("Spheres : %d", sphereCount);
					ImGui::Text("Meshes : %d", meshCount);
					ImGui::SliderInt("Current Object", &objectIndex, 0, sphereCount + meshCount - 1);

					if (objectIndex < sphereCount)
					{
						if (ImGui::SliderFloat3("Position", value_ptr(spheres[objectIndex].position), -100.0f, 100.0f))
							cudaDirty = true;
						if (ImGui::SliderFloat("Radius", &(spheres[objectIndex].radius), EPSILON, 100))
							cudaDirty = true;
						if (ImGui::ListBox("Material Type", (int*)&(spheres[objectIndex].material.type), MATERIAL_TYPE_ARRAY, IM_ARRAYSIZE(MATERIAL_TYPE_ARRAY)))
							cudaDirty = true;
						if (ImGui::SliderFloat3("Color", value_ptr(spheres[objectIndex].material.color), 0.0f, 1.0f))
							cudaDirty = true;
						if (ImGui::SliderFloat3("Emission", value_ptr(spheres[objectIndex].material.emission), 0.0f, 10.0f))
							cudaDirty = true;
					}
					else
					{
						int meshIndex = objectIndex - sphereCount;
						ImGui::Text("Triangles : %d", meshes[meshIndex].count);
						if (ImGui::SliderFloat3("Position", value_ptr(meshes[meshIndex].position), -100.0f, 100.0f))
							cudaDirty = true;
					}
				}
			}
			ImGui::End();
			ImGui::Render();
		}
		glutSwapBuffers();
	}

#pragma endregion Opengl Callbacks

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Cuda Tracer");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW()\n");
		return -1;
	}

	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat position[] = { 0.0f, 40.0f, 0.0f, 0.0f };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
	glMateriali(GL_FRONT, GL_SHININESS, 15);

	glShadeModel(GL_SMOOTH);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glClearColor(0.6, 0.65, 0.85, 0);

	FreeImage_Initialise();

	// imgui
	{
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void) io;

		ImGui_ImplGLUT_Init();

		ImGui::StyleColorsDark();
	}

	// Init

	{
		camera = new Camera;
	}

	{
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &viewGLTexture);
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGraphicsGLRegisterImage(&viewResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &viewResource);
		cudaGraphicsSubResourceGetMappedArray(&viewArray, viewResource, 0, 0);
	}

	{
		FREE_IMAGE_FORMAT fif = FIF_HDR;
		FIBITMAP *src(nullptr);
		FIBITMAP *dst(nullptr);
		BYTE* bits(nullptr);
		float4* cpuHDRmap;

		src = FreeImage_Load(fif, HDR_FILE_NAME);
		dst = FreeImage_ToneMapping(src, FITMO_REINHARD05);
		bits = FreeImage_GetBits(dst);
		if (bits == nullptr)
			return -1;

		cpuHDRmap = new float4[HDRWidth * HDRHeight];

		for (int x = 0; x < HDRWidth; x++)
		{
			for (int y = 0; y < HDRHeight; y++)
			{
				RGBQUAD rgbQuad;
				FreeImage_GetPixelColor(dst, x, y, &rgbQuad);
				cpuHDRmap[y*HDRWidth + x].x = rgbQuad.rgbRed / 256.0f;
				cpuHDRmap[y*HDRWidth + x].y = rgbQuad.rgbGreen / 256.0f;
				cpuHDRmap[y*HDRWidth + x].z = rgbQuad.rgbBlue / 256.0f;
				cpuHDRmap[y*HDRWidth + x].w = 1.0f;
			}
		}

		gpuErrorCheck(cudaMalloc(&cudaHDRmap, HDRWidth * HDRHeight * sizeof(float4)));
		gpuErrorCheck(cudaMemcpy(cudaHDRmap, cpuHDRmap, HDRWidth * HDRHeight * sizeof(float4), cudaMemcpyHostToDevice));

		HDRtexture.filterMode = cudaFilterModeLinear;
		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &HDRtexture, cudaHDRmap, &channel4desc, HDRWidth * HDRHeight * sizeof(float4));

		printf("Load HDR Map Success\n");
		printf("Width : %d\nHeight : %d\n", HDRWidth, HDRHeight);

		FreeImage_Unload(src);
		FreeImage_Unload(dst);
		delete cpuHDRmap;
	}


#if ENABLE_KDTREE
	int mesheCount = sizeof(meshes) / sizeof(Mesh);
	for (int i = 0; i < mesheCount; i++)
	{
		meshes[i].tree = new KDTree(meshes[i].triangles, meshes[i].count);
		meshes[i].tree->Build();
	}
#endif

	glutKeyboardFunc(Keyboard);
	glutKeyboardUpFunc(KeyboardUp);
	glutSpecialFunc(Special);
	glutSpecialUpFunc(SpecialUp);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutMouseWheelFunc(MouseWheel);
	glutMouseFunc(Mouse);
	glutPassiveMotionFunc(Motion);
	glutMotionFunc(Motion);
	glutDisplayFunc(Display);
	glutMainLoop();

	// Cleanup
	cudaDeviceReset();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();
	return 0;
}