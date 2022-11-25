#pragma once
#include "model.h"

const float PI = acos(-1);

static inline void ComputeNormal(int face_id, int x, int y, float& x_coord, float& y_coord, float& z_coord, float length)
{
	length--;
	switch (face_id)
	{
	case 0:   //positive x (right face)
		x_coord = 0.5f;
		y_coord = -0.5f + y / length;
		z_coord = -0.5f + x / length;
		break;
	case 1:   //negative x (left face)		
		x_coord = -0.5f;
		y_coord = -0.5f + y / length;
		z_coord = 0.5f - x / length;
		break;
	case 2:   //positive y (top face)
		x_coord = -0.5f + x / length;
		y_coord = 0.5f;
		z_coord = -0.5f + y / length;
		break;
	case 3:   //negative y (bottom face)
		x_coord = -0.5f + x / length;
		y_coord = -0.5f;
		z_coord = 0.5f - y / length;
		break;
	case 4:   //positive z (back face)
		x_coord = 0.5f - x / length;
		y_coord = -0.5f + y / length;
		z_coord = 0.5f;
		break;
	case 5:   //negative z (front face)
		x_coord = -0.5f + x / length;
		y_coord = -0.5f + y / length;
		z_coord = -0.5f;
		break;
	default:
		break;
	}
}

/*
* 基于拟蒙特卡洛方法的随机低差异序列Hammersley
* 基于Van Der Corput
*/
static inline float RadicalInverse_VDC(unsigned int bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; //  0x100000000
}
//获取大小为N的样本集中的低差异样本i
static inline Eigen::Vector2f Hammersley(unsigned int i, unsigned int n)
{
	return Eigen::Vector2f(float(i) / float(n), RadicalInverse_VDC(i));
}

static inline Eigen::Vector3f ImportanceSampleGGX(Eigen::Vector2f Xi, Eigen::Vector3f N, float roughness)
{
	float a = roughness * roughness;

	float phi = 2.f * PI * Xi.x();
	float cosTheta = sqrt((1.f - Xi.y()) / (1.f + (a * a - 1.f) * Xi.y()));
	float sinTheta = sqrt(1.f - cosTheta * cosTheta);

	Eigen::Vector3f H(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
	Eigen::Vector3f up = abs(N.z()) < 0.999f ? Eigen::Vector3f(0, 0, 1) : Eigen::Vector3f(1, 0, 0);
	Eigen::Vector3f tangent = up.cross(N).normalized();
	Eigen::Vector3f bi = N.cross(tangent);

	Eigen::Vector3f sampleVec = tangent * H.x() + bi * H.y() + N * H.z();
	return sampleVec.normalized();
}

/*
* 根据CubeMap计算辐射度纹理
*/
static inline CubeMap* GenerateIrradianceMap(CubeMap* cubeMap)
{
	//创建辐射度CubeMap
	const int irSize = 64;
	const float PI = acos(-1);
	CubeMap* irradianceMap = new CubeMap(irSize, irSize);
	int ids0 = 0, ids1 = 0, ids2 = 0, ids3 = 0, ids4 = 0, ids5 = 0;
	//遍历CubeMap的每个纹理
	for (int facei = 0; facei < 6; facei++)
	{
		//遍历facei纹理的每个像素
		for (int x = 0; x < irSize; x++)
		{
			for (int y = 0; y < irSize; y++)
			{
				Eigen::Vector3f normal;
				ComputeNormal(facei, x, y, normal.x(), normal.y(), normal.z(), irSize);//计算出法线
				normal.normalize();
				Eigen::Vector3f up = abs(normal.y()) < 0.999f ? Eigen::Vector3f(0, 1, 0) : Eigen::Vector3f(0, 0, 1);
				Eigen::Vector3f right = up.cross(normal).normalized();
				up = normal.cross(right);
				Eigen::Vector3f irradiance(0, 0, 0);

				float sampleDelta = 0.25;
				int numSamples = 0;
				for (float phi = 0.f; phi < 2.f * PI; phi += sampleDelta)
				{
					for (float theta = 0.f; theta < 0.5f * PI; theta += sampleDelta)
					{
						Eigen::Vector3f tanSample(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
						Eigen::Vector3f sampleVec = tanSample.x() * right + tanSample.y() * up + tanSample.z() * normal;
						sampleVec.normalize();
						Eigen::Vector3f col = cubeMap->GetData(sampleVec).head(3);

						irradiance += col * sin(theta) * cos(theta);
						numSamples++;
					}
				}
				irradiance = PI * irradiance * (1.f / numSamples);
				Eigen::Vector4f irr(irradiance.x(), irradiance.y(), irradiance.z(), 1);
				irradianceMap->m_Textures[facei]->SetData(Eigen::Vector2f(1.f * x / irSize, 1.f * y / irSize), irr);
			}
		}
	}
	return irradianceMap;
}

//计算预滤波环境贴图
static inline std::vector<CubeMap*>* GeneratePrefilterMap(CubeMap* cubeMap, int levels)
{
	const int maxLevels = 4;
	std::vector<CubeMap*>* prefilterMaps = new std::vector<CubeMap*>();
	int maxSize = 32;
	levels = std::max(maxLevels, levels);

	//遍历mip-map的每个level
	for (int levelidx = 0; levelidx < levels; levelidx++)
	{
		int factor = 1 << levelidx;
		int size = maxSize / factor;
		CubeMap* preMap = new CubeMap(size, size);
		//遍历每个面
		for (int facei = 0; facei < 6; facei++)
		{
			//遍历每个像素
			for (int x = 0; x < size; x++)
			{
				for (int y = 0; y < size; y++)
				{
					Eigen::Vector3f normal;
					ComputeNormal(facei, x, y, normal.x(), normal.y(), normal.z(), size);
					normal.normalize();
					Eigen::Vector3f up = abs(normal.y()) < 0.999f ? Eigen::Vector3f(0, 1, 0) : Eigen::Vector3f(0, 0, 1);
					Eigen::Vector3f right = up.cross(normal).normalized();
					up = normal.cross(right);

					Eigen::Vector3f r = normal;
					Eigen::Vector3f v = normal;

					Eigen::Vector3f prefilterColor(0, 0, 0);
					float totalWeight = 0;
					int numSamples = 1024;
					for (int i = 0; i < numSamples; i++)
					{
						Eigen::Vector2f Xi = Hammersley(i, numSamples);
						Eigen::Vector3f h = ImportanceSampleGGX(Xi, normal, levelidx / maxLevels);
						Eigen::Vector3f l = (2 * v.dot(h) * h - v).normalized();

						Eigen::Vector3f col = cubeMap->GetData(l).head(3);
						float nDotl = std::max(normal.dot(l), 0.f);

						if (nDotl > 0)
						{
							prefilterColor += col * nDotl;
							totalWeight += nDotl;
						}
					}
					prefilterColor /= totalWeight;
					Eigen::Vector4f pcolor(prefilterColor.x(), prefilterColor.y(), prefilterColor.z(), 1);
					preMap->m_Textures[facei]->SetData(Eigen::Vector2f(1.f * x / size, 1.f * y / size), pcolor);
				}
			}
		}
		prefilterMaps->push_back(preMap);
	}
	return prefilterMaps;
}
