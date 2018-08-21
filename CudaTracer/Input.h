#ifndef H_INPUT
#define H_INPUT

bool keyState[256] = { false };
bool mouseState[3] = { false };
int mousePos[2] = { 0 };

inline bool IsKeyDown(unsigned char key)
{
	return keyState[key];
}

inline bool IsMouseDown(int mouse)
{
	return mouseState[mouse];
}

inline void WarpMouse(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
	glutWarpPointer(x, y);
}

inline void GetMousePos(int &xPos, int &yPos)
{
	xPos = mousePos[0];
	yPos = mousePos[1];
}

#endif