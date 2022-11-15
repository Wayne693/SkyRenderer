#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Draw.h"
#include "Model.h"
#include "Camera.h"
#include "Scene.h"
#include "Shader.h"
#include <stdio.h>
#include <math.h>
#include <iostream>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

extern const int WIDTH;
extern const int HEIGHT;

//模型空间左手系
//世界空间左手系
//观察空间右手系
//裁剪空间左手系
//屏幕空间左手系

//三角重心插值，返回1-u-v,u,v
Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P)
{
	Eigen::Vector3f u = Eigen::Vector3f(B.x() - A.x(), C.x() - A.x(), A.x() - P.x()).cross(Eigen::Vector3f(B.y() - A.y(), C.y() - A.y(), A.y() - P.y()));// u v 1
	return Eigen::Vector3f(1.f - (u.x() + u.y()) / u.z(), u.x() / u.z(), u.y() / u.z());
}

/*
* RenderLoop负责将场景渲染到FrameBuffer上
* 包含颜色与深度信息
*/
static inline void RenderLoop(FrameBuffer* frameBuffer, FrameBuffer* shadowMap, Scene* mainScene, Shader* shader)
{
	auto models = mainScene->GetModels();
	auto camera = (*mainScene->GetCameras())[0];//目前只有一个相机

	//获取当前shader的dataTruck
	DataTruck* dataTruck = &shader->dataTruck;
	dataTruck->camera = camera;
	dataTruck->WIDTH = WIDTH;
	dataTruck->HEIGHT = HEIGHT;
	dataTruck->shadowMap = shadowMap;

	for (int modelIdx = 0; modelIdx < models->size(); modelIdx++)//遍历所有模型
	{
		auto model = (*models)[modelIdx];
		model->UpdateModelMatrix();
		camera->UpdateVPMatrix();
		Eigen::Matrix4f matrixM = model->GetModelMatrix();
		Eigen::Matrix4f matrixVP = camera->GetVPMatrix();
		//向shader传递MVP矩阵,当前model
		dataTruck->matrixM = matrixM;
		dataTruck->matrixVP = matrixVP;
		dataTruck->model = model;
		Light mainLight = mainScene->GetLight();
		dataTruck->mainLight = mainLight;
		auto meshes = model->GetMeshes();

		//遍历每个模型的所有mesh
		for (int meshIdx = 0; meshIdx < meshes->size(); meshIdx++)
		{
			auto mesh = (*meshes)[meshIdx];
			auto pFace = mesh->GetPositionFaces();
			auto nFace = mesh->GetNormalFaces();
			auto vtFace = mesh->GetUVFaces();

			//处理每个三角
			for (int i = 0; i < pFace->size(); i++)
			{
				dataTruck->Clear();
				
				Face positionFace = (*pFace)[i];
				Face normalFace = (*nFace)[i];
				Face uvFace = (*vtFace)[i];

				//加载顶点position
				Eigen::Vector4f posA = (*mesh->GetPositions())[positionFace.A - 1];
				Eigen::Vector4f posB = (*mesh->GetPositions())[positionFace.B - 1];
				Eigen::Vector4f posC = (*mesh->GetPositions())[positionFace.C - 1];
				dataTruck->DTpositionOS.push_back(posA);
				dataTruck->DTpositionOS.push_back(posB);
				dataTruck->DTpositionOS.push_back(posC);
				//加载顶点normal
				dataTruck->DTnormalOS.push_back((*mesh->GetNormals())[normalFace.A - 1]);
				dataTruck->DTnormalOS.push_back((*mesh->GetNormals())[normalFace.B - 1]);
				dataTruck->DTnormalOS.push_back((*mesh->GetNormals())[normalFace.C - 1]);
				//加载顶点uv
				Eigen::Vector2f uvA = (*mesh->GetTexcoords())[uvFace.A - 1];
				Eigen::Vector2f uvB = (*mesh->GetTexcoords())[uvFace.B - 1];
				Eigen::Vector2f uvC = (*mesh->GetTexcoords())[uvFace.C - 1];
				dataTruck->DTuv0.push_back(uvA);
				dataTruck->DTuv0.push_back(uvB);
				dataTruck->DTuv0.push_back(uvC);

				//运行顶点着色器
				shader->Vert();

				//背面剔除
				auto positionCS = dataTruck->DTpositionCS;
				Eigen::Vector3f v1 = (positionCS[1]/positionCS[1].w() - positionCS[0]/positionCS[0].w()).head(3);
				Eigen::Vector3f v2 = (positionCS[2]/positionCS[2].w() - positionCS[0]/positionCS[0].w()).head(3);
				Eigen::Vector3f vNormal = v1.cross(v2);
				if (vNormal.z() <= 0)
				{
					continue;
				}

				//获取三角包围盒
				auto positionSS = dataTruck->DTpositionSS;
				auto a = positionSS[0];
				auto b = positionSS[1];
				auto c = positionSS[2];
				int minx = std::max(0, std::min(WIDTH, (int)std::min(a.x(), std::min(b.x(), c.x()))));
				int miny = std::max(0, std::min(HEIGHT, (int)std::min(a.y(), std::min(b.y(), c.y()))));
				int maxx = std::min(WIDTH, std::max(0, (int)std::max(a.x(), std::max(b.x(), c.x()))));
				int maxy = std::min(HEIGHT, std::max(0, (int)std::max(a.y(), std::max(b.y(), c.y()))));

				
				//遍历包围盒中每个像素
				for (int x = minx; x <= maxx; x++)
				{
					for (int y = miny; y <= maxy; y++)
					{
						//三角插值
						Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));
						//像素在三角形内
						if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
						{
							//插值出深度
							float depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();
							//深度测试
							
							if (depth > frameBuffer->GetZ(x, y))
							{
								continue;
							}

							//运行片元着色器
							auto finalColor = shader->Frag(u.x(), u.y(), u.z());
							DrawPoint(frameBuffer, x, y, finalColor);
 							frameBuffer->SetZ(x, y, depth);
							//std::cout << x << " " << y << " " << depth << std::endl;
						}
					}
				}
			}
		}
	}
}