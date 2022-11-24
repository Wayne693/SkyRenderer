#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

const Eigen::Vector2f invAtan = Eigen::Vector2f(0.1591, 0.3183);
Eigen::Vector2f SampleSphericalMap(Eigen::Vector3f pos)
{
	Eigen::Vector2f uv(atan2(pos.z(), pos.x()), asin(pos.y()));
	uv.x() = uv.x() * invAtan.x() + 0.5;
	uv.y() = uv.y() * invAtan.y() + 0.5;
	return uv;
}

void SkyBoxShader::Vert()
{
	auto matrixM = dataTruck->matrixM;
	auto matrixVP = dataTruck->matrixVP;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck->DTpositionWS.push_back(matrixM * dataTruck->DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck->DTpositionCS.push_back(matrixVP * dataTruck->DTpositionWS[i]);
		//将positionCS转到positionSS
		auto vertex = dataTruck->DTpositionCS[i];
		auto tmp = ComputeScreenPos(vertex);
		dataTruck->DTpositionSS.push_back(tmp);
		//dataTruck->DTpositionSS[i].z() = 1.f;
	}
}

Eigen::Vector4f SkyBoxShader::Frag(float a, float b, float c)
{
	//插值出纹理坐标(透视矫正插值)
	float alpha = a / dataTruck->DTpositionWS[0].z();
	float beta = b / dataTruck->DTpositionWS[1].z();
	float gamma = c / dataTruck->DTpositionWS[2].z();
	float zn = 1 / (alpha + beta + gamma);
	//插值出世界坐标(透视矫正插值)
	Eigen::Vector3f positionOS = zn * (alpha * dataTruck->DTpositionOS[0] + beta * dataTruck->DTpositionOS[1] + gamma * dataTruck->DTpositionOS[2]).head(3);
	Eigen::Vector3f normalOS = a * dataTruck->DTnormalOS[0] + b * dataTruck->DTnormalOS[1] + c * dataTruck->DTnormalOS[2];
	//normalOS.normalize();
	CubeMap* cubeMap = dataTruck->mesh->GetCubeMap();
	//采样CubeMap
	Eigen::Vector4f finalColor = cubeMap->GetData(positionOS.normalized());
	//std::cout << finalColor << std::endl;
	return finalColor;
}