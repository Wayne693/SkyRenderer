#include "Shader.h"
#include <iostream>

void ShadowMapShader::Vert()
{
	auto matrixM = dataTruck.matrixM;
	//auto matrixVP = dataTruck.matrixVP;
	int WIDTH = dataTruck.WIDTH;
	int HEIGHT = dataTruck.HEIGHT;
	Eigen::Vector3f sCameraLookat = dataTruck.mainLight.direction;
	Eigen::Vector3f asixY(0, 1, 0);
	Eigen::Vector3f sCameraAsixX = sCameraLookat.cross(asixY).normalized();
	Eigen::Vector3f sCameraUp = sCameraAsixX.cross(sCameraLookat).normalized();
	Camera sCamera = *dataTruck.camera;
	sCamera.SetLookAt(sCameraLookat);
	sCamera.SetUp(sCameraUp);
	sCamera.UpdateOrthoVPMatrix(); 
	//std::cout << sCamera.GetLookAt() << " " << sCamera.GetUp() << std::endl;
	auto matrixVP = sCamera.GetOrthoVPMatrix();
	
	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck.DTpositionWS.push_back(matrixM * dataTruck.DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck.DTpositionCS.push_back(matrixVP * dataTruck.DTpositionWS[i]);
		//将positionCS转到positionSS
		auto vertex = dataTruck.DTpositionCS[i];
		auto tmp = Eigen::Vector4f(vertex.x() * WIDTH / (2 * vertex.w()) + WIDTH / 2, vertex.y() * HEIGHT / (2 * vertex.w()) + HEIGHT / 2, vertex.z() / vertex.w(), vertex.w());
		dataTruck.DTpositionSS.push_back(tmp);
	}
}

Eigen::Vector4f ShadowMapShader::Frag(float a, float b, float c)
{
	float z = a * dataTruck.DTpositionSS[0].z() + b * dataTruck.DTpositionSS[1].z() + c * dataTruck.DTpositionSS[2].z();
	z = (z + 1.f) / 2;
	Eigen::Vector4f depth(z * 255, z * 255, z * 255, 255);
	Eigen::Vector4f finalColor = depth;
	return finalColor;
}