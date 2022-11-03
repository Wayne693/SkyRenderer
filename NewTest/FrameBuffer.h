#pragma once

class FrameBuffer
{
private:
	unsigned int* m_RawBuffer;
	float* m_ZBuffer;
	unsigned int m_Width;
	unsigned int m_Height;

public:
	FrameBuffer(int width,int height);
	FrameBuffer(int width, int height, unsigned int color);
	~FrameBuffer();

	unsigned int* GetRawBuffer();
	float* GetZbuffer();
	float GetZ(int x,int y);
	unsigned int width();
	unsigned int height();

	void SetColor(int x,int y,unsigned int color);
	void SetZ(int x, int y, float depth);
	void Clear(unsigned int clearColor);//将ColorBuffer清空为clearColor,将ZBuffer清空
};