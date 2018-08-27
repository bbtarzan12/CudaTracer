#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
		position = glm::vec3(4.950867f, 26.793819f, 58.673916f);
		fov = 70.0f;
		nearPlane = 0.1f;
		farPlane = 1000.0f;
		moveSpeed = 25.0f;
		mouseSpeed = 10.0f;
		pitch = -24.790009;
		yaw = 213.499847f;
		view = mat4(0);
		proj = mat4(0);
		aperture = 0;
		focalDistance = 0.1f;
	}

	__device__ Ray GetRay(int x, int y)
	{
		vec3 wDir = glm::normalize(-forward);
		vec3 uDir = glm::normalize(cross(up, wDir));
		vec3 vDir = glm::cross(wDir, -uDir);

		float top = glm::tan(fov * glm::pi<float>() / 360.0f);
		float right = aspectRatio * top;
		float bottom = -top;
		float left = -right;

		float imPlaneUPos = left + (right - left)*((x + 0.5f) / (float)width);
		float imPlaneVPos = bottom + (top - bottom)*((y + 0.5f) / (float)height);

		vec3 originDirection = imPlaneUPos * uDir + imPlaneVPos * vDir - wDir;
		return Ray(position, normalize(originDirection));
	}

	__device__ Ray GetRay(curandState* randState, int x, int y, bool dof)
	{
		float jitterValueX = curand_uniform(randState) - 0.5;
		float jitterValueY = curand_uniform(randState) - 0.5;

		vec3 wDir = glm::normalize(-forward);
		vec3 uDir = glm::normalize(cross(up, wDir));
		vec3 vDir = glm::cross(wDir, -uDir);

		float top = glm::tan(fov * glm::pi<float>() / 360.0f);
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
				float apertureX = cos(angle) * distance;
				float apertureY = sin(angle) * distance;

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
		if (IsMouseDown(0))
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
		toggleMouseMovement = IsMouseDown(0);
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

		pitch = glm::clamp(pitch, -89.0f, 89.0f);
		yaw = glm::mod(yaw, 360.0f);

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
	}
	bool hit;
	float t;
	vec3 normal;
	Material material;
};

struct Triangle
{
	__host__ __device__ Triangle() {}
	__host__ __device__ Triangle(vec3 pos0, vec3 pos1, vec3 pos2, Material material)
	{
		pos[0] = pos0;
		pos[1] = pos1;
		pos[2] = pos2;
		hasTexture = false;
		this->material = material;
	}

	__host__ __device__ Triangle(vec3 pos0, vec3 pos1, vec3 pos2, vec3 tex0, vec3 tex1, vec3 tex2, Material material)
	{
		pos[0] = pos0;
		pos[1] = pos1;
		pos[2] = pos2;
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

		vec3 normal = normalize(cross(pos[1] - pos[0], pos[2] - pos[0]));

		vec3 v0v1 = pos[1] - pos[0];
		vec3 v0v2 = pos[2] - pos[0];
		vec3 pvec = cross(ray.direction, v0v2);
		float det = dot(v0v1, pvec);
		if (fabs(det) < epsilon<float>()) return ObjectIntersection(hit, t, normal, material);

		vec3 tvec = ray.origin - pos[0];
		u = dot(tvec, pvec);
		if (u < 0 || u > det) return ObjectIntersection(hit, t, normal, material);

		vec3 qvec = cross(tvec, v0v1);
		v = dot(ray.direction, qvec);
		if (v < 0 || u + v > det) return ObjectIntersection(hit, t, normal, material);

		t = dot(v0v2, qvec) / det;

		if (t < epsilon<float>()) return ObjectIntersection(hit, t, normal, material);

		hit = true;
		return ObjectIntersection(hit, t, normal, material);
	}

	vec3 pos[3];
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
	__device__ ObjectIntersection Intersect(const Ray &ray) const
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
		return ObjectIntersection(hit, distance, normal, material);
	}
};

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


		std::vector<tinyobj::shape_t> obj_shapes;
		std::vector<tinyobj::material_t> obj_materials;
		std::vector<Material> materials;

		printf("Loading %s...\n", fileName);
		std::string err = tinyobj::LoadObj(obj_shapes, obj_materials, inputFile.c_str(), mtlBasePath.c_str());

		if (!err.empty())
			std::cerr << err << std::endl;

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

		long shapeSize, indicesSize;
		shapeSize = obj_shapes.size();
		std::vector<Triangle>* triangles = new std::vector<Triangle>;

		for (int i = 0; i < shapeSize; i++)
		{
			indicesSize = obj_shapes[i].mesh.indices.size() / 3;
			for (size_t f = 0; f < indicesSize; f++)
			{

				vec3 v0_ = vec3(
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3 + 1],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f] * 3 + 2]
				);

				vec3 v1_ = vec3(
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3 + 1],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 1] * 3 + 2]
				);

				vec3 v2_ = vec3(
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3 + 1],
					obj_shapes[i].mesh.positions[obj_shapes[i].mesh.indices[3 * f + 2] * 3 + 2]
				);

				vec3 t0_, t1_, t2_;

				if (obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1 < obj_shapes[i].mesh.texcoords.size())
				{
					t0_ = vec3(
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2],
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f] * 2 + 1],
						0
					);

					t1_ = vec3(
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2],
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 1] * 2 + 1],
						0
					);

					t2_ = vec3(
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2],
						obj_shapes[i].mesh.texcoords[obj_shapes[i].mesh.indices[3 * f + 2] * 2 + 1],
						0
					);
				}
				else
				{
					t0_ = vec3(0, 0, 0);
					t1_ = vec3(0, 0, 0);
					t2_ = vec3(0, 0, 0);
				}

				Triangle triangle;
				if (obj_shapes[i].mesh.material_ids[f] < materials.size())
				{
					triangle = Triangle(v0_, v1_, v2_, t0_, t1_, t2_, materials[obj_shapes[i].mesh.material_ids[f]]);
				}
				else
				{
					triangle = Triangle(v0_, v1_, v2_, t0_, t1_, t2_, Material());
				}
				triangles->push_back(triangle);
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

	__device__ ObjectIntersection Intersect(Ray ray)
	{
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
		return intersection;
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
		triangles.push_back(Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]], material));
	}

	return Mesh(pos, triangles.data(), 12, material);
}

dim3 block, grid;
Camera* camera;

Sphere spheres[] =
{
	Sphere(vec3(20, 10, 14), 8, Material(TRANS,  vec3(1))),
	Sphere(vec3(-14, 8, -20), 8, Material(DIFF,  vec3(1))),
	Sphere(vec3(-14, 8, 14), 8, Material(SPEC,  vec3(1))),
	Sphere(vec3(14, 8, -14), 8, Material(GLOSS,  vec3(1)))
	//Sphere(vec3(0, 65, 0), 8, Material(DIFF, vec3(0.75, 0.75, 0.75), vec3(2.2, 2.2, 2.2))),
	//Sphere(vec3(0, 30, 0), 8,  Material(TRANS,  vec3(1)))
};
Mesh meshes[] =
{
	//Mesh(vec3(0,0,0), "Cornell_Long.obj")
	Mesh(vec3(0,0,0), "Cornell.obj")
	//Mesh(vec3(0,0,0), "Board.obj")
	//CreateBox(vec3(0, 30, 0), vec3(30, 1, 30), Material(DIFF, vec3(0.75, 0.75, 0.75), vec3(2.2, 2.2, 2.2))),
	//Mesh(vec3(0, 0, 0), "board.obj", Material(DIFF)),
	//Mesh(vec3(0, 3, 0), "Crystal_Low.obj", Material(TRANS)),
	////CreateBox(vec3(0, 0, 0), vec3(30, 1, 30), Material(DIFF, vec3(0.75, 0.75, 0.75))),
	//CreateBox(vec3(30, 15, 0), vec3(1, 15, 30), Material(DIFF, vec3(0.0, 0.0, 0.75))),
	//CreateBox(vec3(-30, 15, 0), vec3(1, 15, 30), Material(DIFF, vec3(0.75, 0.0, 0.0))),
	//CreateBox(vec3(0, 15, 30), vec3(30, 15, 1), Material(DIFF, vec3(0.75, 0.75, 0.75))),
	//CreateBox(vec3(0, 15, -30), vec3(30, 15, 1), Material(DIFF, vec3(0.75, 0.75, 0.75)))
};

#pragma endregion Scene

#pragma region Kernels

__device__ Ray GetReflectedRay(Ray ray, vec3 position, glm::vec3 normal, vec3 &color, Material material, curandState* randState)
{
	switch (material.type)
	{
	case DIFF:
	{
		vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
		float r1 = 2.0f * pi<float>() * curand_uniform(randState);
		float r2 = curand_uniform(randState);
		float r2s = sqrt(r2);

		vec3 w = nl;
		vec3 u;
		if (fabs(w.x) > 0.1f)
			u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
		else
			u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
		vec3 v = cross(w, u);
		vec3 reflected = normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
		color *= material.color;
		return Ray(position, reflected);
	}
	case GLOSS:
	{
		float phi = 2 * pi<float>() * curand_uniform(randState);
		float r2 = curand_uniform(randState);
		float phongExponent = 20;
		float cosTheta = pow(1 - r2, 1.0f / (phongExponent + 1));
		float sinTheta = sin(1 - cosTheta * cosTheta);

		vec3 w = normalize(ray.direction - normal * 2.0f * dot(normal, ray.direction));
		vec3 u = normalize(cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
		vec3 v = cross(w, u);

		vec3 reflected = normalize(u * cos(phi) * sinTheta + v * sin(phi) * sinTheta + w * cosTheta);
		color *= material.color;
		return Ray(position, reflected);
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

		if (cos2t < EPSILON) return Ray(position, reflection);

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
			color *= (RP);
			return Ray(position, reflection);
		}
		color *= (TP);
		return Ray(position, tdir);
	}
	case SPEC:
	{
		vec3 reflected = ray.direction - normal * 2.0f * dot(normal, ray.direction);
		color *= material.color;
		return Ray(position, reflected);
	}
	}
}

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

// Path Tracing
__device__ vec3 TraceRay(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, curandState* randState)
{
	vec3 resultColor = vec3(0, 0, 0);
	vec3 mask = vec3(1, 1, 1);

	for (int depth = 0; depth < MAX_DEPTH; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);

		if (intersection.hit == 0) return resultColor * vec3(0.2f, 0.2f, 0.2f);
		resultColor += mask * intersection.material.emission;
		vec3 position = ray.origin + ray.direction * intersection.t;
		ray = GetReflectedRay(ray, position, intersection.normal, mask, intersection.material, randState);
	}
	return resultColor;
}

// Photon Map (Render Debug Image)
__device__ vec3 TraceRay(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, Photon* map, int maxPhotons)
{
	vec3 color = vec3(0, 0, 0);
	ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);
	if (intersection.hit == false) return vec3(0, 0, 0);
	vec3 position = ray.origin + ray.direction * intersection.t;
	int nearPhotonCount = 0;
	for (int i = 0; i < maxPhotons; i++)
	{
		if (dot(intersection.normal ,map[i].normal) > 0 && distance(position, map[i].position) <= 0.2f)
		{
			color += map[i].power;
			nearPhotonCount++;
		}
	}
	if (nearPhotonCount < 4)
		return vec3(0, 0, 0);
	color /= (float)nearPhotonCount;
	return color;
}

// Photon Map (Building Photon Map)
__device__ Photon TraceRay(Ray ray, vec3 lightEmission, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, curandState* randState)
{
	vec3 resultColor = lightEmission;
	MaterialType beforeType = NONE;
	for (int depth = 0; depth < 4; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);

		if(intersection.hit == false) return Photon();

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
		ray = GetReflectedRay(ray, position, intersection.normal, resultColor, intersection.material, randState);
	}
	return Photon();
}

// Path Tracing + Photon Map
__device__ vec3 TraceRay(Ray ray, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, Photon* map, int maxPhotons, curandState* randState)
{
	vec3 resultColor = vec3(0, 0, 0);
	vec3 mask = vec3(1, 1, 1);

	for (int depth = 0; depth < MAX_DEPTH; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, meshes, sphereCount, meshCount);

		if (intersection.hit == 0)
		{
			float longlatX = atan2(ray.direction.x, ray.direction.z);
			longlatX = longlatX < 0.f ? longlatX + two_pi<float>() : longlatX;
			float longlatY = acos(-ray.direction.y);

			float u = longlatX / two_pi<float>();
			float v = longlatY / pi<float>();

			int u2 = (int) (u * HDRWidth);
			int v2 = (int) (v * HDRHeight);

			int HDRtexelidx = u2 + v2 * HDRWidth;

			float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);
			vec3 HDRcol2 = vec3(HDRcol.x, HDRcol.y, HDRcol.z);

			return resultColor + (mask * HDRcol2);
		}


		vec3 position = ray.origin + ray.direction * intersection.t;
		vec3 emission = intersection.material.emission;
		vec3 photonColor = vec3(0, 0, 0);
		int nearPhotonCount = 0;
		for (int i = 0; i < maxPhotons; i++)
		{
			float dist = distance(position, map[i].position);
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

		resultColor += mask * emission;
		ray = GetReflectedRay(ray, position, intersection.normal, mask, intersection.material, randState);
	}
	return resultColor;
}

// Image Rendering Kernel
__global__ void PathKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int loopX, int loopY, bool dof, vec3* deviceImage)
{
	int width = camera->width;
	int height = camera->height;
	int x = gridDim.x * blockDim.x * loopX + blockIdx.x * blockDim.x + threadIdx.x;
	int y = gridDim.y * blockDim.y * loopY + blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int i = y * width + x;

	if (i >= width * height) return;

	curandState randState;

	float invSample = 1.0f / TRACE_SAMPLES;
	vec3 color = vec3(0, 0, 0);
	for (int s = 0; s < TRACE_SAMPLES; s++)
	{
		curand_init(WangHash(threadId) + WangHash(s), 0, 0, &randState);
		Ray ray = camera->GetRay(&randState, x, y, dof);
		color += TraceRay(ray, spheres, meshes, sphereCount, meshCount, &randState);
	}
	color *= invSample;
	deviceImage[i] = color;
}

// Real Time Rendering Kernel
__global__ void PathKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int loopX, int loopY, bool dof, int frame, cudaSurfaceObject_t surface)
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
	vec3 color = TraceRay(ray, spheres, meshes, sphereCount, meshCount, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float)(frame - 1) + color) / (float)frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

// Real time + Photon Mapping Kernel
__global__ void PathKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int loopX, int loopY, bool dof, int frame, Photon* map, int mapSize, cudaSurfaceObject_t surface)
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
	vec3 color = TraceRay(ray, spheres, meshes, sphereCount, meshCount, map, mapSize, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float)(frame - 1) + color) / (float)frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

// Photon Mapping Debug Image Rendering Kernel
__global__ void DebugPhotonMapKernel(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, Photon* map, int maxPhotons, vec3* deviceImage)
{
	int width = camera->width;
	int height = camera->height;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int i = y * width + x;
	if (i >= width * height) return;

	Ray ray = camera->GetRay(x, y);
	vec3 color = TraceRay(ray, spheres, meshes, sphereCount, meshCount, map, maxPhotons);
	deviceImage[i] = color;
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
		vec3 dir = normalize(vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)));
		Ray ray = Ray(lightPos, dir);
		photon = TraceRay(ray, lightEmission, spheres, meshes, sphereCount, meshCount, &randState);
	}
	map[i] = photon;
}

// Image Rendering Loop
void TracingLoop(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, bool dof, vec3* deviceImage)
{
	int progress = 0;
	for (int i = 0; i < TRACE_OUTER_LOOP_X; i++)
	{
		for (int j = 0; j < TRACE_OUTER_LOOP_Y; j++)
		{
			cudaEvent_t start, stop;
			float elapsedTime;
			cudaEventCreate(&start);
			cudaEventRecord(start, 0);
			PathKernel<< <grid, block >> > (camera, spheres, meshes, sphereCount, meshCount, i, j, dof, deviceImage);
			cudaEventCreate(&stop);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&elapsedTime, start, stop);
			printf("\rTracing %d/%d  |  Elapsed time : %f ms", ++progress, TRACE_OUTER_LOOP_X * TRACE_OUTER_LOOP_Y, elapsedTime);
			cudaDeviceSynchronize();
		}
	}
}

// Real Time Rendering Loop
void TracingLoop(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int frame, bool dof, cudaSurfaceObject_t surface)
{
	int progress = 0;
	for (int i = 0; i < TRACE_OUTER_LOOP_X; i++)
	{
		for (int j = 0; j < TRACE_OUTER_LOOP_Y; j++)
		{
			cudaEvent_t start, stop;
			float elapsedTime;
			cudaEventCreate(&start);
			cudaEventRecord(start, 0);
			PathKernel << <grid, block >> > (camera, spheres, meshes, sphereCount, meshCount, i, j, dof, frame, surface);
			cudaEventCreate(&stop);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&elapsedTime, start, stop);
			printf("\rTracing %d/%d  |  Elapsed time : %f ms", ++progress, TRACE_OUTER_LOOP_X * TRACE_OUTER_LOOP_Y, elapsedTime);
			cudaDeviceSynchronize();
		}
	}
}

// Photon Mapping Rendering Loop
void TracingLoop(Camera* camera, Sphere* spheres, Mesh* meshes, int sphereCount, int meshCount, int frame, bool dof, Photon* map, int mapSize, cudaSurfaceObject_t surface)
{
	int progress = 0;
	for (int i = 0; i < TRACE_OUTER_LOOP_X; i++)
	{
		for (int j = 0; j < TRACE_OUTER_LOOP_Y; j++)
		{
			cudaEvent_t start, stop;
			float elapsedTime;
			cudaEventCreate(&start);
			cudaEventRecord(start, 0);
			PathKernel << <grid, block >> > (camera, spheres, meshes, sphereCount, meshCount, i, j, dof, frame, map, mapSize, surface);
			cudaEventCreate(&stop);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&elapsedTime, start, stop);
			printf("\rTracing %d/%d  |  Elapsed time : %f ms", ++progress, TRACE_OUTER_LOOP_X * TRACE_OUTER_LOOP_Y, elapsedTime);
			cudaDeviceSynchronize();
		}
	}
}

void SaveImage(const char* path, int width, int height, const vec3* colors)
{
	FIBITMAP *dib = FreeImage_Allocate(width, height, 32);
	RGBQUAD rgb;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			int i = y * width + x;
			rgb.rgbRed = std::fmax(std::fmin(colors[i].r * 255, 255), 0);
			rgb.rgbGreen = std::fmax(std::fmin(colors[i].g * 255, 255), 0);
			rgb.rgbBlue = std::fmax(std::fmin(colors[i].b * 255, 255), 0);
			rgb.rgbReserved = 1.0f;
			FreeImage_SetPixelColor(dib, x, height - y, &rgb);
		}
	}
	dib = FreeImage_ConvertTo24Bits(dib);
	bool success = FreeImage_Save(FIF_PNG, dib, path);
	if (!success) std::cout << "Image Save Error " << std::endl;
	else std::cout << "Image Save Success : " << path << std::endl;
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

	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memoryAllocTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	PhotonMapKernel << <grid, block >> > (cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, lightPos, lightEmission, maxPhotons, cudaPhotonMap, frame);
	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&renderingTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Building Photon Map End | Memory Allocation Time : %f ms | Building time : %f ms\n", memoryAllocTime, renderingTime);

	{
		//int width = camera->width;
	//int height = camera->height;
	//vec3* deviceImage;
	//gpuErrorCheck(cudaMalloc(&deviceImage, width * height * sizeof(vec3)));

	//gpuErrorCheck(cudaMemcpy(cudaDebugPhotonMap, cudaPhotonMap, sizeof(Photon) * maxPhotons, cudaMemcpyDeviceToDevice));
	//block = dim3(16, 9);
	//grid.x = ceil(width / block.x);
	//grid.y = ceil(height / block.y);

	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);
	//DebugPhotonMapKernel<<<grid, block>>>(cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, cudaDebugPhotonMap, maxPhotons, deviceImage);
	//cudaDeviceSynchronize();
	//cudaEventCreate(&stop);
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&renderingTime, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//printf("Calculate Debug Photon Map Finished : %f ms\n", renderingTime);

	//vec3* hostImage = new vec3[width * height];
	//gpuErrorCheck(cudaMemcpy(hostImage, deviceImage, width * height * sizeof(vec3), cudaMemcpyDeviceToHost));
	//SaveImage("Result_PhotonMap.png", width, height, hostImage);
	////photons = new Photon[maxPhotons];
	////gpuErrorCheck(cudaMemcpy(photons, cudaDebugPhotonMap, sizeof(Photon) * maxPhotons, cudaMemcpyDeviceToHost));

	//cudaFree(deviceImage);
	//delete hostImage;
	//cudaFree(cudaDebugPhotonMap);
	}

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

void RenderImage(bool dof)
{
	int width = camera->width;
	int height = camera->height;

	block = dim3(32, 18);
	grid.x = ceil(ceil(width / TRACE_OUTER_LOOP_X) / block.x);
	grid.y = ceil(ceil(height / TRACE_OUTER_LOOP_Y) / block.y);

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

	vec3* deviceImage;
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaMalloc(&deviceImage, width * height * sizeof(vec3));
	TracingLoop(cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, dof, deviceImage);
	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\rRendering End | Elapsed time : %f ms\n", elapsedTime);

	vec3* hostImage = new vec3[width * height];
	cudaMemcpy(hostImage, deviceImage, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);

	SaveImage("Result.png", width, height, hostImage);
	cudaFree(deviceImage);
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
	delete hostImage;
}

void RenderRealTime(cudaSurfaceObject_t surface, bool dof, bool photon, int frame)
{
	int width = camera->width;
	int height = camera->height;

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

	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memoryAllocTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	int photonMapSize = 0;
	Photon* cudaPhotonMap;
	Photon* photonMap;
	if (photon)
	{
		photonMapSize = 100000;
		photonMap = BuildPhotonMap(photonMapSize, frame);

		gpuErrorCheck(cudaMalloc(&cudaPhotonMap, sizeof(Photon) * photonMapSize));
		gpuErrorCheck(cudaMemcpy(cudaPhotonMap, photonMap, sizeof(Photon) * photonMapSize, cudaMemcpyHostToDevice));
	}

	block = dim3(16, 9);
	grid.x = ceil(ceil(width / TRACE_OUTER_LOOP_X) / block.x);
	grid.y = ceil(ceil(height / TRACE_OUTER_LOOP_Y) / block.y);

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	TracingLoop(cudaCamera, cudaSpheres, cudaMeshes, sphereCount, meshCount, frame, dof, cudaPhotonMap, photonMapSize, surface);
	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&renderingTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("\rRendering End(%d) | Memory Allocation Time : %f ms | Rendering time : %f ms\n", frame, memoryAllocTime, renderingTime);

	if (photon)
	{
		delete photonMap;
		cudaFree(cudaPhotonMap);
	}
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
				frame = 1;
				cudaDirty = false;
			}
			RenderRealTime(viewCudaSurfaceObject, enableDof, enablePhoton, frame++);
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

		if (enableSaveImage && frame >= TRACE_SAMPLES)
		{
			enableSaveImage = false;
			cudaToggle = false;
			cudaDirty = false;
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
				float lat0 = pi<float>() * (-float(0.5) + (float)(i - 1) / lats);
				float z0 = radius * sin(lat0);
				float zr0 = radius * cos(lat0);

				float lat1 = pi<float>() * (-float(0.5) + (float)i / lats);
				float z1 = radius * sin(lat1);
				float zr1 = radius * cos(lat1);

				glBegin(GL_QUAD_STRIP);
				for (j = 0; j <= longs; j++)
				{
					float lng = 2 * pi<float>() * (float)(j - 1) / longs;
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
			}
			glPopMatrix();
		}
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
	cudaDeviceReset();
	return 0;
}