#include "Draw.h"
#include "DataPool.h"


__device__ void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color = white)
{
	auto width = frameBuffer->m_Width;
	auto height = frameBuffer->m_Height;
	color *= 255;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		CudaSetBufferData(frameBuffer->m_ColorBufferID, pos, Vector4fToColor(normalColor(color)));
	}
}

__device__ void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth)
{
	auto width = frameBuffer->m_Width;
	auto height = frameBuffer->m_Height;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		CudaSetBufferData(frameBuffer->m_ZBufferID, pos, *(uint32_t*)&depth);
	}
}