#pragma once
#include "Dense"

class FrameBuffer
{
public:
	int m_ColorBufferID;
	int m_ZBufferID;
	unsigned int m_Width;
	unsigned int m_Height;

	FrameBuffer();
	FrameBuffer(int width,int height);
	FrameBuffer(int width, int height, unsigned int color);

	unsigned int* GetRawBuffer();
	float* GetZbuffer();
	unsigned int width();
	unsigned int height();

	//��ColorBuffer���ΪclearColor,��ZBuffer����Ϊ1
	void Clear(unsigned int clearColor);
};