#pragma once
#include "Dense"
#include "Model.h"
#include "DataPool.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
extern const int WIDTH;
extern const int HEIGHT;

//将uv坐标平移缩放
__host__ __device__ static Eigen::Vector2f TransformTex(Eigen::Vector2f uv, Texture* texture)
{
	float x = uv.x() * texture->m_Tilling.x() + texture->m_Offset.x();
	float y = uv.y() * texture->m_Tilling.y() + texture->m_Offset.y();
	return Eigen::Vector2f(x, y);
}

//根据uv坐标采样纹理
__host__ __device__ static Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv)
{
	//std::cout << "texture data address = " << textureRawData.data() << std::endl;
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
		uint32_t val = GetRawData(texture->m_ID, pos);
		//printf("id = %d pos = %d \n", m_ID, pos);
		uint8_t mask = 255;
		Eigen::Vector4f data(val & mask, (val >> 8) & mask, (val >> 16) & mask, (val >> 24) & mask);
		data /= 255.f;
		return data;
	}
	return Eigen::Vector4f(0, 0, 0, 0);
	//return texture->GetData(uv);
}

__host__ __device__ static Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv)
{
	Eigen::Vector4f data = Tex2D(normalTexture, uv);
	return 2 * data.head(3) - Eigen::Vector3f(1, 1, 1);
}

__host__ __device__ static Eigen::Vector4f ComputeScreenPos(Eigen::Vector4f positionCS)
{
	return Eigen::Vector4f(positionCS.x() * WIDTH / (2 * positionCS.w()) + WIDTH / 2, positionCS.y() * HEIGHT / (2 * positionCS.w()) + HEIGHT / 2, positionCS.z() / positionCS.w(), positionCS.w());
}