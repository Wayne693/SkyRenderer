#include "Shader.h"
#include <iostream>

void ShadowMapShader::Vert()
{
	auto matrixM = dataTruck->matrixM;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;
	Eigen::Vector3f sCameraLookat = dataTruck->mainLight.direction.normalized();

	Eigen::Vector3f asixY(0, 1, 0);
	Eigen::Vector3f sCameraAsixX = sCameraLookat.cross(asixY).normalized();
	Eigen::Vector3f sCameraUp = sCameraAsixX.cross(sCameraLookat).normalized();
	std::vector<Eigen::Vector3f>* visualCone = dataTruck->camera->GetVisualCone();

	Camera sCamera = *(dataTruck->camera);
	sCamera.SetLookAt(sCameraLookat);
	sCamera.SetUp(sCameraUp);
	sCamera.UpdateViewMatrix(); 

	Eigen::Matrix4f matrixV = sCamera.GetViewMatrix();

	float minx, maxx, miny, maxy, minz, maxz;
	//transform visual cone from worldspace to lightspace
	for (int i = 0; i < visualCone->size(); i++)
	{
		(*visualCone)[i] = matrixV.block(0, 0, 3, 3) * (*visualCone)[i];
		if (i == 0)
		{
			minx = (*visualCone)[i].x();
			maxx = minx;
			miny = (*visualCone)[i].y();
			maxy = miny;
			minz = (*visualCone)[i].z();
			maxz = minz;
		}
		else
		{
			minx = std::min(minx, (*visualCone)[i].x());
			maxx = std::max(maxx, (*visualCone)[i].x());
			miny = std::min(miny, (*visualCone)[i].y());
			maxy = std::max(maxy, (*visualCone)[i].y());
			minz = std::min(minz, (*visualCone)[i].z());
			maxz = std::max(maxz, (*visualCone)[i].z());
		}
	}
	Eigen::Vector4f center((minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2, 1);
	center = matrixV.inverse() * center;
	center = center / center.w();
	sCamera.SetSize((maxy - miny) / 2);
	sCamera.SetAspect((maxx - minx) / (maxy - miny));
	auto cminz = minz;
	minz = -maxz;
	maxz = -cminz;
	sCamera.SetFarPlane(maxz);
	sCamera.SetNearPlane(minz);
	sCamera.SetPosition(center.head(3) - sCameraLookat * ((maxz - minz) / 2 + minz));
	sCamera.UpdateOrthoVPMatrix();
	auto matrixVP = sCamera.GetOrthoVPMatrix();
	matrixV = sCamera.GetViewMatrix();
	auto matrixP = sCamera.GetOrthoMatrix();
	//将lightMatrixVP赋值给dataTruck，供后面的渲染使用
	dataTruck->lightMatrixVP = matrixVP;
	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck->DTpositionWS.push_back(matrixM * dataTruck->DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck->DTpositionCS.push_back(matrixVP * dataTruck->DTpositionWS[i]);
	}
}

Eigen::Vector4f ShadowMapShader::Frag(Face face, float a, float b, float c)
{
	float z = a* dataTruck->DTpositionSS[face.A].z() + b * dataTruck->DTpositionSS[face.B].z() + c * dataTruck->DTpositionSS[face.C].z();
	z = (z + 1.f) / 2;
	Eigen::Vector4f depth(z, z, z, 1);
	Eigen::Vector4f finalColor = depth;
	return finalColor;
}