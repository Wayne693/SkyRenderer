#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Draw.h"
#include "Model.h"
#include "Camera.h"
#include "Scene.h"
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

std::vector<Eigen::Vector4f> positionWS;//世界空间的顶点坐标
std::vector<Eigen::Vector4f> positionCS;//裁剪空间的顶点坐标
std::vector<Eigen::Vector4f> positionSS;//屏幕空间的顶点坐标
std::vector<Eigen::Vector3f> normalWS;	//世界空间法线

Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P)//三角重心插值，返回1-u-v,u,v
{
	Eigen::Vector3f u = Eigen::Vector3f(B.x() - A.x(), C.x() - A.x(), A.x() - P.x()).cross(Eigen::Vector3f(B.y() - A.y(), C.y() - A.y(), A.y() - P.y()));// u v 1
	return Eigen::Vector3f(1.f - (u.x() + u.y()) / u.z(), u.x() / u.z(), u.y() / u.z());
}

Eigen::Vector4f mulColor(Eigen::Vector4f a, Eigen::Vector4f b)//颜色相乘
{
	return Eigen::Vector4f(a.x() * b.x() / 255.0, a.y() * b.y() / 255.0, a.z() * b.z() / 255.0, a.w() * b.w() / 255.0);
}


static inline void RenderLoop(GLuint renderTexture, FrameBuffer* frameBuffer, Scene* mainScene)
{
	ImDrawList* drawList = ImGui::GetBackgroundDrawList();

	auto models = mainScene->GetModels();
	auto camera = (*mainScene->GetCameras())[0];//目前只有一个相机

	for (int i = 0; i < models->size(); i++)//遍历所有模型
	{
		auto model = (*models)[i];	
		model->UpdateModelMatrix();
		camera->UpdateVPMatrix();
		Eigen::Matrix4f matrixM = model->GetModelMatrix();
		Eigen::Matrix4f matrixVP = camera->GetVPMatrix();

		auto p = model->GetPositions();
		for (int i = 0; i < p->size(); i++)//处理顶点
		{
			Eigen::Vector4f tmp = (*p)[i];
			tmp = matrixM * tmp;//model->world
			positionWS.push_back(tmp);
			tmp = matrixVP * tmp;//world->clip
			positionCS.push_back(tmp);
			//tmp /= tmp.w();//转为NDC空间
			//转为屏幕空间
			tmp = Eigen::Vector4f(tmp.x() * WIDTH / (2 * tmp.w()) + WIDTH / 2, tmp.y() * HEIGHT / (2 * tmp.w()) + HEIGHT / 2, tmp.z() / tmp.w(), tmp.w());
			positionSS.push_back(tmp);
		}

		auto pn = model->GetNormals();
		for (int i = 0; i < pn->size(); i++)
		{
			Eigen::Vector3f tmp = (*pn)[i];
			Eigen::Vector4f ttmp(tmp.x(), tmp.y(), tmp.z(), 0);
			ttmp = matrixM.inverse().transpose() * ttmp;//将法线转为世界空间
			normalWS.push_back(ttmp.head(3));
		}

		auto puv = model->GetTexcoords();

		auto pFace = model->GetPositionFaces();
		auto nFace = model->GetNormalFaces();
		auto vtFace = model->GetUVFaces();
		for (int i = 0; i < pFace->size(); i++)//处理片元
		{
			Eigen::Vector4f a, b, c;
			Face positionFace = (*pFace)[i];
			Face normalFace = (*nFace)[i];
			Face uvFace = (*vtFace)[i];
			a = positionSS[positionFace.A - 1];
			b = positionSS[positionFace.B - 1];
			c = positionSS[positionFace.C - 1];



			//背面剔除
			Eigen::Vector3f worldPos = (positionWS[positionFace.A - 1].head(3) + positionWS[positionFace.B - 1].head(3) + positionWS[positionFace.C - 1].head(3)) / 3;
			Eigen::Vector3f worldViewDir = camera->GetPosition() - worldPos;
			worldViewDir.normalize();
			Eigen::Vector3f v1 = (positionWS[positionFace.B - 1] - positionWS[positionFace.A - 1]).head(3);
			Eigen::Vector3f v2 = (positionWS[positionFace.C - 1] - positionWS[positionFace.A - 1]).head(3);
			Eigen::Vector3f vNormal = v1.cross(v2);
			vNormal.normalize();
			if (worldViewDir.dot(vNormal) < 0)
			{
				continue;
			}
			//DrawLine(frameBuffer, a.x(), a.y(), b.x(), b.y(), white);
			//DrawLine(frameBuffer, b.x(), b.y(), c.x(), c.y(), white);
			//DrawLine(frameBuffer, a.x(), a.y(), c.x(), c.y(), white);

			//获取三角包围盒
			int minx = std::max(0, std::min(WIDTH, (int)std::min(a.x(), std::min(b.x(), c.x()))));
			int miny = std::max(0, std::min(HEIGHT, (int)std::min(a.y(), std::min(b.y(), c.y()))));
			int maxx = std::min(WIDTH, std::max(0, (int)std::max(a.x(), std::max(b.x(), c.x()))));
			int maxy = std::min(HEIGHT, std::max(0, (int)std::max(a.y(), std::max(b.y(), c.y()))));

			Light mainLight = mainScene->GetLight();
			auto lightDir = mainLight.direction;
			auto lightColor = mainLight.color;
			lightDir.normalize();

			for (int x = minx; x <= maxx; x++)
			{
				for (int y = miny; y <= maxy; y++)
				{
					Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));
					if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)//像素在三角形内
					{
						float depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();//插值出深度

						if (depth > frameBuffer->GetZ(x, y))
						{
							continue;
						}
						//插值出法线
						Eigen::Vector3f normal = u.x() * normalWS[normalFace.A - 1] + u.y() * normalWS[normalFace.B - 1] + u.z() * normalWS[normalFace.C - 1];
						//插值出纹理坐标
						Eigen::Vector2f uv = u.x() * (*puv)[uvFace.A - 1] + u.y() * (*puv)[uvFace.B - 1] + u.z() * (*puv)[uvFace.C - 1];
						//目前只有一个texture,todo:处理多个texture
						Texture* tp = (*model->GetTextures())[0];
						//tilling、offset后的uv坐标
						Eigen::Vector2f TFuv =Eigen::Vector2f(tp->GetTilling().x() * uv.x() * tp->width(), tp->GetTilling().y() * uv.y() * tp->height()) + tp->GetOffset();
						//std::cout << TFuv << std::endl;
						normal.normalize();
						
						float NdotL = normal.dot(lightDir);
						frameBuffer->SetZ(x, y, depth);
						Eigen::Vector4f finalColor = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(lightColor, tp->GetData(TFuv));
						//std::cout << lightColor  << std::endl;
						DrawPoint(frameBuffer, x, y, finalColor);
						
					}
				}
			}
		}
	}

	ImTextureID imguiId = (ImTextureID)renderTexture;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameBuffer->width(), frameBuffer->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, frameBuffer->GetRawBuffer());
	drawList->AddImage(imguiId, ImVec2(0, 0), ImVec2(WIDTH, HEIGHT));
	
	//数据清空
	frameBuffer->Clear(Vector4fToColor(black));
	positionWS.clear();
	positionCS.clear();
	positionSS.clear();
	normalWS.clear();
}