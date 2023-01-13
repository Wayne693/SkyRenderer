#pragma once
#include "Dense"
#include "thrust/extrema.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//常用颜色
const Eigen::Vector4f white = Eigen::Vector4f(255, 255, 255, 255);
const Eigen::Vector4f black = Eigen::Vector4f(0, 0, 0, 255);
const Eigen::Vector4f red = Eigen::Vector4f(255, 0, 0, 255);
const Eigen::Vector4f green = Eigen::Vector4f(0, 255, 0, 255);
const Eigen::Vector4f blue = Eigen::Vector4f(0, 0, 255, 255);

__host__ __device__ static uint32_t Vector4fToColor(Eigen::Vector4f color)
{
	unsigned int rtnColor = 0;
	//小端CPU颜色编码
	rtnColor |= static_cast<uint8_t>(color.w()) << 24;
	rtnColor |= static_cast<uint8_t>(color.z()) << 16;
	rtnColor |= static_cast<uint8_t>(color.y()) << 8;
	rtnColor |= static_cast<uint8_t>(color.x()) << 0;
	return rtnColor;
}

__host__ __device__ static Eigen::Vector4f ColorToVector4f(unsigned int color)
{
	return Eigen::Vector4f(color & 255, (color >> 8) & 255, (color >> 16) & 255, (color >> 24) & 255);
}

//将颜色分量截断在(0,255)
__device__ static Eigen::Vector4f normalColor(Eigen::Vector4f color)
{
	return Eigen::Vector4f(thrust::min(255.f, thrust::max(0.f, color.x())), thrust::min(255.f, thrust::max(0.f, color.y())), thrust::min(255.f, thrust::max(0.f, color.z())), thrust::min(255.f, thrust::max(0.f, color.w())));
}

//颜色相乘
__host__ __device__ static Eigen::Vector4f Vec4Mul(Eigen::Vector4f a, Eigen::Vector4f b)
{
	return Eigen::Vector4f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

__host__ __device__ static Eigen::Vector3f Vec3Mul(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}