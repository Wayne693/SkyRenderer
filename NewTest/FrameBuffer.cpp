#include "FrameBuffer.h"
#include "LowLevelAPI.h"
#include "DataPool.h"
#include <iostream>

const float initDepth = 1;

FrameBuffer::FrameBuffer()
{
	m_ColorBufferID = -1;
	m_ZBufferID = -1;
	m_Height = 0;
	m_Width = 0;
}

FrameBuffer::FrameBuffer(int width, int height):m_Width(width),m_Height(height)
{
	//m_RawBuffer = (unsigned int*)malloc(sizeof(unsigned int) * width * height);
	//m_ZBuffer = (float*)malloc(sizeof(float) * width * height);
	m_ColorBufferID = AddBufferData(nullptr, width * height);
	m_ZBufferID = AddBufferData(nullptr, width * height);
	if (m_ColorBufferID == -1 || m_ZBufferID == -1)
	{
		std::cout << "Create frameBuffer Failed!" << std::endl;
	}
	//std::fill_n(m_ZBuffer, m_Height * m_Width, initDepth);
	ClearBuffer(m_ZBufferID, m_Height * m_Width, initDepth);
}

FrameBuffer::FrameBuffer(int width, int height, unsigned int color) :m_Width(width), m_Height(height)
{
	/*m_RawBuffer = (unsigned int*)malloc(sizeof(unsigned int) * width * height);
	m_ZBuffer = (float*)malloc(sizeof(float) * width * height);*/
	m_ColorBufferID = AddBufferData(nullptr, width * height);
	m_ZBufferID = AddBufferData(nullptr, width * height);

	if (m_ColorBufferID == -1 || m_ZBufferID == -1)
	{
		std::cout << "Create frameBuffer Failed!" << std::endl;
	}
	else
	{
		ClearBuffer(m_ZBufferID, m_Height * m_Width, initDepth);
		ClearBuffer(m_ColorBufferID, m_Height * m_Width, color);
	}
}

FrameBuffer::~FrameBuffer()
{

}

unsigned int* FrameBuffer::GetRawBuffer()
{
	if (m_ColorBufferID != -1)
		return GetBuffer(m_ColorBufferID);
	return nullptr;
}

float* FrameBuffer::GetZbuffer()
{
	if (m_ZBufferID != -1)
		return (float *) GetBuffer(m_ZBufferID);
	return nullptr;
}

float FrameBuffer::GetZ(int x, int y)
{
	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		//return m_ZBuffer[pos];
		uint32_t depth = GetBufferData(m_ZBufferID, pos);
		//将uint32_t按字节转为float
		float rt = *(float*)&depth;
		return rt;
	}
	return 1;
}

Eigen::Vector4f FrameBuffer::GetRaw(int x, int y)
{
	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		//return ColorToVector4f(m_RawBuffer[pos]);
		return ColorToVector4f(GetBufferData(m_ColorBufferID, pos));
	}
	return black;
}

void FrameBuffer::SetColor(int x, int y, unsigned int color)
{
	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		//m_RawBuffer[pos] = color;
		SetBufferData(m_ColorBufferID, pos, color);
	}
}

void FrameBuffer::SetZ(int x,int y, float depth)
{
	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		//m_ZBuffer[pos] = depth;
		SetBufferData(m_ZBufferID, pos, *(uint32_t*)&depth);
	}
}

unsigned int FrameBuffer::height()
{
	return m_Height;
}

unsigned int FrameBuffer::width()
{
	return m_Width;
}

void FrameBuffer::Clear(unsigned int clearColor)
{
	ClearBuffer(m_ColorBufferID, m_Height * m_Width, clearColor);
	ClearBuffer(m_ZBufferID, m_Height * m_Width, initDepth);
}