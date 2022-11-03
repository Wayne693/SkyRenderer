#include "FrameBuffer.h"
#include <iostream>
#include <stdlib.h>

const float initDepth = 1;

FrameBuffer::FrameBuffer(int width, int height):m_Width(width),m_Height(height)
{
	m_RawBuffer = (unsigned int*)malloc(sizeof(unsigned int) * width * height);
	m_ZBuffer = (float*)malloc(sizeof(float) * width * height);
	
	if (m_RawBuffer == nullptr || m_ZBuffer == nullptr)
	{
		std::cout << "Create frameBuffer Failed!" << std::endl;
	}
	std::fill_n(m_ZBuffer, m_Height * m_Width, initDepth);
}

FrameBuffer::FrameBuffer(int width, int height, unsigned int color) :m_Width(width), m_Height(height)
{
	m_RawBuffer = (unsigned int*)malloc(sizeof(unsigned int) * width * height);
	m_ZBuffer = (float*)malloc(sizeof(float) * width * height);
	
	if (m_RawBuffer == nullptr || m_ZBuffer == nullptr)
	{
		std::cout << "Create frameBuffer Failed!" << std::endl;
	}
	else
	{
		std::fill_n(m_ZBuffer, m_Height * m_Width, initDepth);
		for (int i = 0; i < width * height; i++)
		{
			m_RawBuffer[i] = color;
		}
	}
}

FrameBuffer::~FrameBuffer()
{
	if (m_RawBuffer)
	{
		free(m_RawBuffer);
	}
	if (m_ZBuffer)
	{
		free(m_ZBuffer);
	}
}

unsigned int* FrameBuffer::GetRawBuffer()
{
	if (m_RawBuffer)
		return m_RawBuffer;
	return nullptr;
}

float* FrameBuffer::GetZbuffer()
{
	if (m_ZBuffer)
		return m_ZBuffer;
	return nullptr;
}

float FrameBuffer::GetZ(int x, int y)
{
	int pos = (m_Height - y) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		return m_ZBuffer[pos];
	}
}

void FrameBuffer::SetColor(int x, int y, unsigned int color)
{
	int pos = (m_Height - y) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		m_RawBuffer[pos] = color;
	}
}

void FrameBuffer::SetZ(int x,int y, float depth)
{
	int pos = (m_Height - y) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Height * m_Width)
	{
		m_ZBuffer[pos] = depth;
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
	for (int i = 0; i < m_Height * m_Width; i++)
	{
		m_RawBuffer[i] = clearColor;
	}
	std::fill_n(m_ZBuffer, m_Height * m_Width, initDepth);
}