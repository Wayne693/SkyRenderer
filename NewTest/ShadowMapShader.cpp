#include "Shader.h"
#include <iostream>


Varyings ShadowMapShader::Vert(Attributes vertex)
{
	auto matrixM = vertex.matrixM;
	auto mainLight = dataTruck->mainLight;
	auto camera = dataTruck->camera;

	Eigen::Vector3f sCameraLookat = mainLight.direction.normalized();

	Eigen::Vector3f asixY(0, 1, 0);
	Eigen::Vector3f sCameraAsixX = sCameraLookat.cross(asixY).normalized();
	Eigen::Vector3f sCameraUp = sCameraAsixX.cross(sCameraLookat).normalized();
	std::vector<Eigen::Vector3f> visualCone = *camera->GetVisualCone();

	Camera sCamera = *camera;
	sCamera.SetLookAt(sCameraLookat);
	sCamera.SetUp(sCameraUp);
	sCamera.UpdateViewMatrix();

	Eigen::Matrix4f matrixV = sCamera.GetViewMatrix();
	float minx, maxx, miny, maxy, minz, maxz;
	//transform visual cone from worldspace to lightspace
	for (int i = 0; i < visualCone.size(); i++)
	{
		visualCone[i] = matrixV.block(0, 0, 3, 3) * visualCone[i];
		if (i == 0)
		{
			minx = visualCone[i].x();
			maxx = minx;
			miny = visualCone[i].y();
			maxy = miny;
			minz = visualCone[i].z();
			maxz = minz;
		}
		else
		{
			minx = std::min(minx, visualCone[i].x());
			maxx = std::max(maxx, visualCone[i].x());
			miny = std::min(miny, visualCone[i].y());
			maxy = std::max(maxy, visualCone[i].y());
			minz = std::min(minz, visualCone[i].z());
			maxz = std::max(maxz, visualCone[i].z());
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
	//将lightMatrixVP赋值给dataTruck，供后面的渲染使用
	dataTruck->lightMatrixVP = matrixVP;


	Varyings o;

	//将positionOS转到positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = dataTruck->lightMatrixVP * o.positionWS;
	
	return o;
}

Eigen::Vector4f ShadowMapShader::Frag(Varyings i)
{
	float z = i.positionCS.z();
	z = (z + 1.f) / 2;
	Eigen::Vector4f depth(z, z, z, 1);
	Eigen::Vector4f finalColor = depth;
	return finalColor;
}