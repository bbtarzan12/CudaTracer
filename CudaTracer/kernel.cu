#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <freeimage.h>

#include "kernel.cuh"
#include "Input.h"

using namespace glm;

enum MaterialType { DIFF, GLOSS, TRANS, SPEC, EMIT };

int oldTimeSinceStart = 0;
float deltaTime = 0;


#pragma region Structs

struct Ray
{
	vec3 origin;
	vec3 direction;
	__host__ __device__ Ray(vec3 origin, vec3 direction)
	{
		this->origin = origin;
		this->direction = direction;
	}
};

struct Camera
{
	__host__ __device__ Camera()
	{
		proj = glm::mat4(1.0f);
		position = glm::vec3(2, 2, 2);
		fov = 70.0f;
		nearPlane = 0.1f;
		farPlane = 1000.0f;
		moveSpeed = 50.0f;
		mouseSpeed = 3.0f;
		pitch = -37.0f;
		yaw = 330.0f;
	}

	__device__ Ray GetRay(curandState* randState, int x, int y)
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

		return Ray(position, (glm::normalize(imPlaneUPos * uDir + imPlaneVPos * vDir - wDir)));
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

		view = lookAt(position, position + forward, up);
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
	__host__ __device__ Material(MaterialType type = DIFF, vec3 color = vec3(0), vec3 emission = vec3(1))
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

struct Sphere
{
	__host__ __device__ Sphere(float radius = 0, vec3 position = vec3(0), Material material = Material())
	{
		this->radius = radius;
		this->position = position;
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

		if (det < 0)
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

#pragma endregion Structs

#pragma region Scene Variables

Camera* camera;
Sphere spheres[] =
{
	Sphere(10000, vec3(0, 10040, 0),Material(EMIT, vec3(1), vec3(3.3f, 3.3f, 3.3f))),
	Sphere(10000, vec3(0, -10010, 0), Material(DIFF, vec3(0.75f, 0.75f, 0.75f))),
	Sphere(10000, vec3(10040, 0, 0),Material(DIFF, vec3(0.75f, 0.25f, 0.25f))),
	Sphere(10000, vec3(-10040, 0, 0), Material(DIFF, vec3(0.25f, 0.25f, 0.75f))),
	Sphere(10000, vec3(0, 0, 10040), Material(DIFF, vec3(0.75f, 0.75f, 0.75f))),
	Sphere(10000, vec3(0, 0, -10040), Material(DIFF, vec3(0.75f, 0.75f, 0.75f))),
	Sphere(10, vec3(0, 10, 0), Material(TRANS, vec3(0.75f, 0.75f, 0.75f)))
};

#pragma endregion Scene Variables

#pragma region Kernels

__device__ Ray GetReflectedRay(Ray ray, vec3 position, glm::vec3 normal, vec3 &color, Material material, curandState* randState)
{
	switch (material.type)
	{
	case DIFF:
	{
		vec3 nl = dot(normal, ray.direction) < 0.0f ? normal : normal * -1.0f;
		float r1 = 2.0f * pi<float>() * curand_uniform(randState);
		float r2 = curand_uniform(randState);
		float r2s = sqrt(r2);

		vec3 w = nl;
		vec3 u;
		if (fabs(w.x) > 0.1)
			u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
		else
			u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
		vec3 v = cross(w, u);
		vec3 reflected = normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
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

		return Ray(position, reflected);
	}
	case TRANS:
	{
		vec3 nl = dot(normal, ray.direction) < 0 ? normal : normal * -1.0f;
		vec3 reflection = ray.direction - normal * 2.0f * dot(normal, ray.direction);
		bool into = dot(normal, nl) > 0;
		float nc = 1.0f;
		float nt = 1.5f;
		float nnt = into ? nc / nt : nt / nc;

		float Re, RP, TP, Tr;
		vec3 tdir = vec3(0.0f, 0.0f, 0.0f);

		float ddn = dot(ray.direction, nl);
		float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

		if (cos2t < 0) return Ray(position, reflection);

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
			color = color * (RP);
			return Ray(position, reflection);
		}

		color = color * (TP);
		return Ray(position, tdir);
	}
	case SPEC:
	{
		vec3 reflected = ray.direction - normal * 2.0f * dot(normal, ray.direction);
		return Ray(position, reflected);
	}
	}
}

__device__ ObjectIntersection Intersect(Ray ray, Sphere* spheres, int count)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp = ObjectIntersection();

	for (int i = 0; i < count; i++)
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
	return intersection;
}

__device__ vec3 TraceRay(Ray ray, Sphere* spheres, int count, curandState* randState)
{
	vec3 resultColor = vec3(1, 1, 1);

	for (int depth = 0; depth < MAX_DEPTH; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, count);

		if (intersection.hit == 0) return vec3(0, 0, 0);

		if (intersection.material.type == EMIT)
			return resultColor * intersection.material.emission;

		vec3 color = intersection.material.color;

		if (depth > ROULETTE_DEPTH)
		{
			float maxReflection = color.x > color.y && color.x > color.z ? color.x : color.y > color.z ? color.y : color.z;
			float random = curand_uniform(randState);

			if (random >= maxReflection)
				return resultColor;

			color /= maxReflection;
		}
		vec3 position = ray.origin + ray.direction * intersection.t;
		ray = GetReflectedRay(ray, position, intersection.normal, color, intersection.material, randState);
		resultColor *= color;
	}
	return resultColor;
}


__global__ void PathKernel(Camera* camera, Sphere* spheres, int count, vec3* deviceImage)
{
	int width = camera->width;
	int height = camera->height;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int i = y * width + x;

	if (i >= width * height) return;

	curandState randState;

	float invSample = 1.0f / TRACE_SAMPLES;
	vec3 color = vec3(0, 0, 0);
	for (int s = 0; s < TRACE_SAMPLES; s++)
	{
		curand_init(threadId + WangHash(s), 0, 0, &randState);
		Ray ray = camera->GetRay(&randState, x, y);
		color += TraceRay(ray, spheres, count, &randState);
	}
	color *= invSample;
	deviceImage[i] = color;
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
}

void Render()
{
	int width = camera->width;
	int height = camera->height;

	dim3 block, grid;
	block = dim3(16, 9);
	grid.x = ceil(ceil(width) / block.x);
	grid.y = ceil(ceil(height) / block.y);

	Camera* cudaCamera;
	gpuErrorCheck(cudaMalloc(&cudaCamera, sizeof(Camera)));
	gpuErrorCheck(cudaMemcpy(cudaCamera, camera, sizeof(Camera), cudaMemcpyHostToDevice));

	int sphereCount = sizeof(spheres) / sizeof(Sphere);
	Sphere* cudaSpheres;
	gpuErrorCheck(cudaMalloc(&cudaSpheres, sizeof(Sphere) * sphereCount));
	gpuErrorCheck(cudaMemcpy(cudaSpheres, spheres, sizeof(Sphere) * sphereCount, cudaMemcpyHostToDevice));

	vec3* deviceImage;
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaMalloc(&deviceImage, width * height * sizeof(vec3));
	PathKernel << <grid, block >> > (cudaCamera, cudaSpheres, sphereCount, deviceImage);
	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\rRendering End | Elapsed time : %f ms\n", elapsedTime);

	vec3* hostImage = new vec3[width * height];
	cudaMemcpy(hostImage, deviceImage, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);

	SaveImage("Result.png", width, height, hostImage);
}

#pragma endregion Kernels

#pragma region Opengl Callbacks

void Keyboard(unsigned char key, int x, int y)
{
	keyState[key] = true;
	mousePos[0] = x;
	mousePos[1] = y;

	if (IsKeyDown('q'))
	{
		Render();
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

	camera = new Camera;

	glutKeyboardFunc(Keyboard);
	glutKeyboardUpFunc(KeyboardUp);
	glutSpecialFunc(Special);
	glutSpecialUpFunc(SpecialUp);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutMouseFunc(Mouse);
	glutPassiveMotionFunc(Motion);
	glutMotionFunc(Motion);
	glutDisplayFunc(Display);
	glutMainLoop();
	return 0;
}