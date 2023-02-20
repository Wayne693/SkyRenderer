#pragma once
#include <Dense>
#include "cuda_runtime.h"
#include "DataPool.h"
#include "thrust/extrema.h"

__host__ __device__
inline uint32_t Vector4fToColor(Eigen::Vector4f color)
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
inline Eigen::Vector4f ColorToVector4f(unsigned int color)
{
	return Eigen::Vector4f(color & 255, (color >> 8) & 255, (color >> 16) & 255, (color >> 24) & 255);
}

//将颜色分量截断在(0,255)
__device__
inline Eigen::Vector4f NormalColor(Eigen::Vector4f color)
{
	return Eigen::Vector4f(thrust::min(255.f, thrust::max(0.f, color.x())), thrust::min(255.f, thrust::max(0.f, color.y())), thrust::min(255.f, thrust::max(0.f, color.z())), thrust::min(255.f, thrust::max(0.f, color.w())));
}

//颜色相乘
__host__ __device__
inline Eigen::Vector4f Vec4Mul(Eigen::Vector4f a, Eigen::Vector4f b)
{
	return Eigen::Vector4f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

__host__ __device__
inline Eigen::Vector3f Vec3Mul(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

__host__ __device__
inline int SelectCubeMapFace(Eigen::Vector3f direction, Eigen::Vector2f* texcoord)
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

__device__
inline bool ZTestAutomic(FrameBuffer* buffer, float depth, int x, int y)
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