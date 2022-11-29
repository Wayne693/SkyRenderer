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
	auto matrixP = dataTruck->camera->GetProjectionMatrix();
	Eigen::Matrix4f matrixV = Eigen::Matrix4f::Zero();
	matrixV << dataTruck->camera->GetViewMatrix().block(0, 0, 3, 3);
	matrixV(3, 3) = 1;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck->DTpositionWS.push_back(matrixM * dataTruck->DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck->DTpositionCS.push_back(matrixP * matrixV * dataTruck->DTpositionWS[i]);
	}
}

Eigen::Vector4f SkyBoxShader::Frag(Face face, float a, float b, float c)
{
	//插值出纹理坐标(透视矫正插值)
	float alpha = a / dataTruck->DTpositionCS[face.A].w();
	float beta = b / dataTruck->DTpositionCS[face.B].w();
	float gamma = c / dataTruck->DTpositionCS[face.C].w();
	float zn = 1 / (alpha + beta + gamma);
	//插值出世界坐标(透视矫正插值)
	Eigen::Vector3f positionWS = zn * (alpha * dataTruck->DTpositionWS[face.A] + beta * dataTruck->DTpositionWS[face.B] + gamma * dataTruck->DTpositionWS[face.C]).head(3);

	CubeMap* cubeMap = dataTruck->mesh->GetCubeMap();
	//采样CubeMap
	Eigen::Vector4f finalColor = cubeMap->GetData(positionWS.normalized());
	//std::cout << finalColor << std::endl;
	return finalColor;
}