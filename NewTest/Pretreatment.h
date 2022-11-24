#pragma once
#include "model.h"

static inline void ComputeNormal(int face_id, int x, int y, float& x_coord, float& y_coord, float& z_coord, float length)
{
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
				Eigen::Vector3f up = normal.y() < 0.999f ? Eigen::Vector3f(0, 1, 0) : Eigen::Vector3f(0, 0, 1);
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
				Eigen::Vector4f irr(irradiance.x(), irradiance.y(), irradiance.z(), 255);
				irradianceMap->m_Textures[facei]->SetData(Eigen::Vector2f(1.f * x / irSize, 1.f * y / irSize), irr);
			}
		}
	}
	return irradianceMap;
}