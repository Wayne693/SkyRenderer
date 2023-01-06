#pragma once
#include "Dense"

class FrameBuffer
{
public:
	//unsigned int* m_RawBuffer;	//Color Buffer
	//float* m_ZBuffer;			//Depth Buffer
	int m_ColorBufferID;
	int m_ZBufferID;
	unsigned int m_Width;
	unsigned int m_Height;

	FrameBuffer();
	FrameBuffer(int width,int height);
	FrameBuffer(int width, int height, unsigned int color);
	~FrameBuffer();

	unsigned int* GetRawBuffer();
	float* GetZbuffer();
	float GetZ(int x,int y);
	Eigen::Vector4f GetRaw(int x, int y);
	unsigned int width();
	unsigned int height();

	void SetColor(int x,int y,unsigned int color);
	void SetZ(int x, int y, float depth);
	void Clear(unsigned int clearColor);//将ColorBuffer清空为clearColor,将ZBuffer重置为1
};