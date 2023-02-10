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

	//将ColorBuffer清空为clearColor,将ZBuffer重置为clearColor
	void Clear(unsigned int clearColor);
};