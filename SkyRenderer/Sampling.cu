#include "Sampling.h"
#include "Utility.h"

__host__ __device__
Eigen::Vector2f TransformTex(Eigen::Vector2f uv, Texture* texture)
{
	float x = uv.x() * texture->m_Tilling.x() + texture->m_Offset.x();
	float y = uv.y() * texture->m_Tilling.y() + texture->m_Offset.y();
	return Eigen::Vector2f(x, y);
}

__device__
Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv)
{
	auto width = texture->m_Width;
	auto height = texture->m_Height;

	int x = (int)(uv.x() * width);
	int y = (int)(uv.y() * height);
	if (x > 0)
	{
		x = x % width;
	}
	else if (x < 0)
	{
		x = width + x % width;
	}

	if (y > 0)
	{
		y = y % height;
	}
	else if (y < 0)
	{
		y = height + y % height;
	}

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < width * height)
	{
		uint32_t val = CudaGetTexData(texture->m_ID, pos);
		uint8_t mask = 255;
		Eigen::Vector4f data(val & mask, (val >> 8) & mask, (val >> 16) & mask, (val >> 24) & mask);
		data /= 255.f;
		return data;
	}
	return Eigen::Vector4f(0, 0, 0, 0);
}

__device__
Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv)
{
	Eigen::Vector4f data = Tex2D(normalTexture, uv);
	return 2 * data.head(3) - Eigen::Vector3f(1, 1, 1);
}

__device__
Eigen::Vector4f ComputeScreenPos(FrameBuffer* frameBuffer, Eigen::Vector4f positionCS)
{
	return Eigen::Vector4f(positionCS.x() * frameBuffer->m_Width / (2 * positionCS.w()) + frameBuffer->m_Width / 2, positionCS.y() * frameBuffer->m_Height / (2 * positionCS.w()) + frameBuffer->m_Height / 2, positionCS.z() / positionCS.w(), positionCS.w());
}

__device__
float GetZ(FrameBuffer* buffer, int x, int y)
{
	auto height = buffer->m_Height;
	auto width = buffer->m_Width;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		uint32_t depth = CudaGetBufferData(buffer->m_ZBufferID, pos);
		float rt = *(float*)&depth;
		return rt;
	}
	return 1;
}

__device__
Eigen::Vector4f CubeMapGetData(CubeMap* cubeMap, Eigen::Vector3f dir)
{
	Eigen::Vector2f uv;
	int idx = SelectCubeMapFace(dir, &uv);
	Texture* tmp = &cubeMap->px;
	switch (idx)
	{
	case 0:
		break;
	case 1:
		tmp = &cubeMap->nx;
		break;
	case 2:
		tmp = &cubeMap->py;
		break;
	case 3:
		tmp = &cubeMap->ny;
		break;
	case 4:
		tmp = &cubeMap->pz;
		break;
	case 5:
		tmp = &cubeMap->nz;
		break;
	}
	return Tex2D(tmp, uv);
}

__device__ 
void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color)
{
	auto width = frameBuffer->m_Width;
	auto height = frameBuffer->m_Height;
	color *= 255;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		CudaSetBufferData(frameBuffer->m_ColorBufferID, pos, Vector4fToColor(NormalColor(color)));
	}
}

__device__ 
void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth)
{
	auto width = frameBuffer->m_Width;
	auto height = frameBuffer->m_Height;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		CudaSetBufferData(frameBuffer->m_ZBufferID, pos, *(uint32_t*)&depth);
	}
}



