#include "Sampling.h"
#include "DataPool.h"
#include "thrust/extrema.h"

__host__ __device__
int selectCubeMapFace(Eigen::Vector3f direction, Eigen::Vector2f* texcoord)
{
	float abs_x = (float)fabs(direction.x());
	float abs_y = (float)fabs(direction.y());
	float abs_z = (float)fabs(direction.z());
	float ma, sc, tc;
	int face_index;

	if (abs_x > abs_y && abs_x > abs_z) {   /* major axis -> x */
		ma = abs_x;
		if (direction.x() > 0) {                  /* positive x */
			face_index = 0;
			sc = -direction.z();
			tc = -direction.y();
		}
		else {                                /* negative x */
			face_index = 1;
			sc = +direction.z();
			tc = -direction.y();
		}
	}
	else if (abs_y > abs_z) {             /* major axis -> y */
		ma = abs_y;
		if (direction.y() > 0) {                  /* positive y */
			face_index = 2;
			sc = +direction.x();
			tc = +direction.z();
		}
		else {                                /* negative y */
			face_index = 3;
			sc = +direction.x();
			tc = -direction.z();
		}
	}
	else {                                /* major axis -> z */
		ma = abs_z;
		if (direction.z() > 0) {                  /* positive z */
			face_index = 4;
			sc = +direction.x();
			tc = -direction.y();
		}
		else {                                /* negative z */
			face_index = 5;
			sc = -direction.x();
			tc = -direction.y();
		}
	}

	texcoord->x() = (sc / ma + 1) / 2;
	texcoord->y() = 1 - (tc / ma + 1) / 2;
	return face_index;
}

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
bool ZTestAutomic(FrameBuffer* buffer, float depth, int x, int y)
{
	auto height = buffer->m_Height;
	auto width = buffer->m_Width;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		return depth == MinZAutomic(buffer->m_ZBufferID, pos, depth);
	}
	return false;
}

__device__
Eigen::Vector4f CubeMapGetData(CubeMap* cubeMap, Eigen::Vector3f dir)
{
	Eigen::Vector2f uv;
	int idx = selectCubeMapFace(dir, &uv);
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
void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color = white)
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

__host__ __device__  
uint32_t Vector4fToColor(Eigen::Vector4f color)
{
	unsigned int rtnColor = 0;
	//小端CPU颜色编码
	rtnColor |= static_cast<uint8_t>(color.w()) << 24;
	rtnColor |= static_cast<uint8_t>(color.z()) << 16;
	rtnColor |= static_cast<uint8_t>(color.y()) << 8;
	rtnColor |= static_cast<uint8_t>(color.x()) << 0;
	return rtnColor;
}

__host__ __device__  
Eigen::Vector4f ColorToVector4f(unsigned int color)
{
	return Eigen::Vector4f(color & 255, (color >> 8) & 255, (color >> 16) & 255, (color >> 24) & 255);
}

//将颜色分量截断在(0,255)
__device__  
Eigen::Vector4f normalColor(Eigen::Vector4f color)
{
	return Eigen::Vector4f(thrust::min(255.f, thrust::max(0.f, color.x())), thrust::min(255.f, thrust::max(0.f, color.y())), thrust::min(255.f, thrust::max(0.f, color.z())), thrust::min(255.f, thrust::max(0.f, color.w())));
}

//颜色相乘
__host__ __device__ 
Eigen::Vector4f Vec4Mul(Eigen::Vector4f a, Eigen::Vector4f b)
{
	return Eigen::Vector4f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

__host__ __device__ 
Eigen::Vector3f Vec3Mul(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}