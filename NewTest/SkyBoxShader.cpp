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

Varyings SkyBoxShader::Vert(Attributes vertex)
{
	auto matrixM = vertex.matrixM;
	auto matrixP = dataTruck->camera->GetProjectionMatrix();
	Eigen::Matrix4f matrixV = Eigen::Matrix4f::Zero();
	matrixV << dataTruck->camera->GetViewMatrix().block(0, 0, 3, 3);
	matrixV(3, 3) = 1;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;
	Varyings o;
	
	//��positionOSת��positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//��positionWSת��positionCS
	o.positionCS = matrixP * matrixV * o.positionWS;
	return o;
}

Eigen::Vector4f SkyBoxShader::Frag(Varyings)
{
	//CubeMap* cubeMap = dataTruck->mesh->GetCubeMap();
	////����CubeMap
	//Eigen::Vector4f finalColor = cubeMap->GetData(positionWS.normalized());
	////std::cout << finalColor << std::endl;
	//return finalColor;
	return Eigen::Vector4f(0, 0, 0, 0);
}