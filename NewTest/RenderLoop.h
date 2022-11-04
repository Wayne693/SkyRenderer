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

//ģ�Ϳռ�����ϵ
//����ռ�����ϵ
//�۲�ռ�����ϵ
//�ü��ռ�����ϵ
//��Ļ�ռ�����ϵ

Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P)//�������Ĳ�ֵ������1-u-v,u,v
{
	Eigen::Vector3f u = Eigen::Vector3f(B.x() - A.x(), C.x() - A.x(), A.x() - P.x()).cross(Eigen::Vector3f(B.y() - A.y(), C.y() - A.y(), A.y() - P.y()));// u v 1
	return Eigen::Vector3f(1.f - (u.x() + u.y()) / u.z(), u.x() / u.z(), u.y() / u.z());
}

static inline void RenderLoop(GLuint renderTexture, FrameBuffer* frameBuffer, Scene* mainScene, Shader* shader)
{
	ImDrawList* drawList = ImGui::GetBackgroundDrawList();

	auto models = mainScene->GetModels();
	auto camera = (*mainScene->GetCameras())[0];//Ŀǰֻ��һ�����

	//��ȡ��ǰshader��dataTruck
	DataTruck* dataTruck = &shader->dataTruck;
	dataTruck->camera = camera;
	dataTruck->WIDTH = WIDTH;
	dataTruck->HEIGHT = HEIGHT;

	for (int i = 0; i < models->size(); i++)//��������ģ��
	{
		auto model = (*models)[i];	
		model->UpdateModelMatrix();
		camera->UpdateVPMatrix();
		Eigen::Matrix4f matrixM = model->GetModelMatrix();
		Eigen::Matrix4f matrixVP = camera->GetVPMatrix();
		//��shader����MVP����,��ǰmodel
		dataTruck->matrixM = matrixM;
		dataTruck->matrixVP = matrixVP;
		dataTruck->model = model;
		

		auto pFace = model->GetPositionFaces();
		auto nFace = model->GetNormalFaces();
		auto vtFace = model->GetUVFaces();
		for (int i = 0; i < pFace->size(); i++)//����ÿ������
		{
			shader->Clear();
			//Eigen::Vector4f a, b, c;
			Face positionFace = (*pFace)[i];
			Face normalFace = (*nFace)[i];
			Face uvFace = (*vtFace)[i];
			
			
			//���ض���position
			dataTruck->DTpositionOS.push_back((*model->GetPositions())[positionFace.A - 1]);
			dataTruck->DTpositionOS.push_back((*model->GetPositions())[positionFace.B - 1]);
			dataTruck->DTpositionOS.push_back((*model->GetPositions())[positionFace.C - 1]);
			//���ض���normal
			dataTruck->DTnormalOS.push_back((*model->GetNormals())[normalFace.A - 1]);
			dataTruck->DTnormalOS.push_back((*model->GetNormals())[normalFace.B - 1]);
			dataTruck->DTnormalOS.push_back((*model->GetNormals())[normalFace.C - 1]);
			//���ض���uv
			dataTruck->DTuv.push_back((*model->GetTexcoords())[uvFace.A - 1]);
			dataTruck->DTuv.push_back((*model->GetTexcoords())[uvFace.B - 1]);
			dataTruck->DTuv.push_back((*model->GetTexcoords())[uvFace.C - 1]);

			//���ж�����ɫ��
			shader->Vert();

			//�����޳�
			auto positionWS = dataTruck->DTpositionWS;
			Eigen::Vector3f worldPos = (positionWS[0].head(3) + positionWS[1].head(3) + positionWS[2].head(3)) / 3;
			Eigen::Vector3f worldViewDir = camera->GetPosition() - worldPos;
			worldViewDir.normalize();
			Eigen::Vector3f v1 = (positionWS[1] - positionWS[0]).head(3);
			Eigen::Vector3f v2 = (positionWS[2] - positionWS[0]).head(3);
			Eigen::Vector3f vNormal = v1.cross(v2);
			vNormal.normalize();
			if (worldViewDir.dot(vNormal) < 0)
			{
				continue;
			}
			
			//��ȡ���ǰ�Χ��
			auto positionSS = dataTruck->DTpositionSS;
			auto a = positionSS[0];
			auto b = positionSS[1];
			auto c = positionSS[2];
			int minx = std::max(0, std::min(WIDTH, (int)std::min(a.x(), std::min(b.x(), c.x()))));
			int miny = std::max(0, std::min(HEIGHT, (int)std::min(a.y(), std::min(b.y(), c.y()))));
			int maxx = std::min(WIDTH, std::max(0, (int)std::max(a.x(), std::max(b.x(), c.x()))));
			int maxy = std::min(HEIGHT, std::max(0, (int)std::max(a.y(), std::max(b.y(), c.y()))));

			Light mainLight = mainScene->GetLight();
			dataTruck->mainLight = mainLight;

			for (int x = minx; x <= maxx; x++)
			{
				for (int y = miny; y <= maxy; y++)
				{
					//���ǲ�ֵ
					Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));
					//��������������
					if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
					{
						//��ֵ�����
						float depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();

						//��Ȳ���
						if (depth > frameBuffer->GetZ(x, y))
						{
							continue;
						}

						//����ƬԪ��ɫ��
						auto finalColor = shader->Frag(u.x(), u.y(), u.z());
						DrawPoint(frameBuffer, x, y, finalColor);
						frameBuffer->SetZ(x, y, depth);
					}
				}
			}
		}
	}

	ImTextureID imguiId = (ImTextureID)renderTexture;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameBuffer->width(), frameBuffer->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, frameBuffer->GetRawBuffer());
	drawList->AddImage(imguiId, ImVec2(0, 0), ImVec2(WIDTH, HEIGHT));
	
	//�������
	frameBuffer->Clear(Vector4fToColor(black));
}