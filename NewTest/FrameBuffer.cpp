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
	m_ColorBufferID = AddBufferData(nullptr, width * height);
	m_ZBufferID = AddBufferData(nullptr, width * height);
	if (m_ColorBufferID == -1 || m_ZBufferID == -1)
	{
		std::cout << "Create frameBuffer Failed!" << std::endl;
	}
	ClearBuffer(m_ZBufferID, m_Height * m_Width, initDepth);
}

FrameBuffer::FrameBuffer(int width, int height, unsigned int color) :m_Width(width), m_Height(height)
{
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