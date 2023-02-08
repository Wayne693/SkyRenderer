#pragma once
#include "Dense"
#include "Model.h"
#include "FrameBuffer.h"
#include "DataPool.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
extern const int WIDTH;
extern const int HEIGHT;

//todo understand
__host__ __device__
inline int selectCubeMapFace(Eigen::Vector3f direction, Eigen::Vector2f* texcoord) {
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

//将uv坐标平移缩放
__host__ __device__ static Eigen::Vector2f TransformTex(Eigen::Vector2f uv, Texture* texture)
{
	float x = uv.x() * texture->m_Tilling.x() + texture->m_Offset.x();
	float y = uv.y() * texture->m_Tilling.y() + texture->m_Offset.y();
	return Eigen::Vector2f(x, y);
}

//根据uv坐标采样纹理
__device__ static Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv)
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
		uint32_t val = CudaGetRawData(texture->m_ID, pos);
		//printf("id = %d pos = %d \n", m_ID, pos);
		uint8_t mask = 255;
		Eigen::Vector4f data(val & mask, (val >> 8) & mask, (val >> 16) & mask, (val >> 24) & mask);
		data /= 255.f;
		return data;
	}
	return Eigen::Vector4f(0, 0, 0, 0);
}

__device__ static Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv)
{
	Eigen::Vector4f data = Tex2D(normalTexture, uv);
	return 2 * data.head(3) - Eigen::Vector3f(1, 1, 1);
}

__device__ static Eigen::Vector4f ComputeScreenPos(FrameBuffer* frameBuffer, Eigen::Vector4f positionCS)
{
	return Eigen::Vector4f(positionCS.x() * frameBuffer->m_Width / (2 * positionCS.w()) + frameBuffer->m_Width / 2, positionCS.y() * frameBuffer->m_Height / (2 * positionCS.w()) + frameBuffer->m_Height / 2, positionCS.z() / positionCS.w(), positionCS.w());
}

//采样FrameBuffer的ZBuffer
__device__ static float GetZ(FrameBuffer* buffer, int x, int y)
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

__device__ static bool ZTestAutomic(FrameBuffer* buffer, float depth, int x, int y)
{
	auto height = buffer->m_Height;
	auto width = buffer->m_Width;

	int pos = (height - y - 1) * width + x;
	if (x >= 0 && x < width && y >= 0 && y < height && pos >= 0 && pos < height * width)
	{
		//float rtdepth = 

		////printf("rtdepth = %lf input_depth = %lf\n", rtdepth, depth);

		//if (rtdepth == depth)
		//{
		//	return true;
		//}
		return depth == MinZAutomic(buffer->m_ZBufferID, pos, depth);
	}
	return false;
}

//采样CubeMap
__device__ 
inline Eigen::Vector4f CubeMapGetData(CubeMap* cubeMap, Eigen::Vector3f dir)
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