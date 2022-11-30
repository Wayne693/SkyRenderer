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
const int ERRORNUM = 1e-5;
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

bool CheckClip(Eigen::Vector4f pos)
{
	if (pos.x() >= -pos.w() && pos.x() <= pos.w() && pos.y() >= -pos.w() && pos.y() <= pos.w() && pos.z() >= -pos.w() && pos.z() <= pos.w())
	{
		return true;
	}
	return false;
}

enum ClipPlane
{
	ERROR,
	RIGHT,
	LEFT,
	TOP,
	BOTTOM,
	NEAR,
	FAR
};

/*
* 渲染状态
* 表明一次渲染的目的
*/
enum RenderConfig
{
	RENDER_SHADOW,
	RENDER_BY_PASS
};

static int IsInsidePlane(ClipPlane plane, Eigen::Vector4f vertex)
{
	switch (plane)
	{
	case ERROR:
		return vertex.w() >= ERRORNUM;
	case RIGHT:
		return vertex.x() <= vertex.w();
	case LEFT:
		return vertex.x() >= -vertex.w();
	case TOP:
		return vertex.y() <= vertex.w();
	case BOTTOM:
		return vertex.y() >= -vertex.w();
	case NEAR:
		return vertex.z() <= vertex.w();
	case FAR:
		return vertex.z() >= -vertex.w();
	default:
		return 0;
	}
}

//计算插值比例
static float GetIntersectRatio(Eigen::Vector4f prev, Eigen::Vector4f curv, ClipPlane plane)
{
	switch (plane)
	{
	case ERROR:
		return (prev.w() + ERRORNUM) / (prev.w() - curv.w());
	case RIGHT:
		return (prev.w() - prev.x()) / ((prev.w() - prev.x()) - (curv.w() - curv.x()));
	case LEFT:
		return (prev.w() + prev.x()) / ((prev.w() + prev.x()) - (curv.w() + curv.x()));
	case TOP:
		return (prev.w() - prev.y()) / ((prev.w() - prev.y()) - (curv.w() - curv.y()));
	case BOTTOM:
		return (prev.w() + prev.y()) / ((prev.w() + prev.y()) - (curv.w() + curv.y()));
	case NEAR:
		return (prev.w() - prev.z()) / ((prev.w() - prev.z()) - (curv.w() - curv.z()));
	case FAR:
		return (prev.w() + prev.z()) / ((prev.w() + prev.z()) - (curv.w() + curv.z()));
	default:
		return 0;
	}
}
//插值
template<class T>
T Lerp(T x, T y, float ratio)
{
	return x + ratio * (y - x);
}
//与一个面裁剪
static void ClipWithPlane(ClipPlane plane, DataTruck* dataTruck)
{
	std::vector<Eigen::Vector4f> tmpPositionCS;
	std::vector<Eigen::Vector4f> tmpPositionWS;
	std::vector<Eigen::Vector3f> tmpNormalWS;
	std::vector<Eigen::Vector2f> tmpUV;
	int size = dataTruck->DTpositionCS.size();
	for (int i = 0; i < size; i++)
	{
		int curi = i;
		int prei = (i - 1 + size) % size;
		Eigen::Vector4f curVet = dataTruck->DTpositionCS[curi];
		Eigen::Vector4f preVet = dataTruck->DTpositionCS[prei];

		int curin = IsInsidePlane(plane, curVet);
		int prein = IsInsidePlane(plane, preVet);
		//两顶点在平面两侧
		if (curin != prein)
		{
			float ratio = GetIntersectRatio(preVet, curVet, plane);

			tmpPositionCS.push_back(Lerp(preVet, curVet, ratio));
			tmpPositionWS.push_back(Lerp(dataTruck->DTpositionWS[prei], dataTruck->DTpositionWS[curi], ratio));
			if (!dataTruck->DTnormalWS.empty())
				tmpNormalWS.push_back(Lerp(dataTruck->DTnormalWS[prei], dataTruck->DTnormalWS[curi], ratio));
			if (!dataTruck->DTuv0.empty())
				tmpUV.push_back(Lerp(dataTruck->DTuv0[prei], dataTruck->DTuv0[curi], ratio));
		}

		if (curin)
		{
			tmpPositionCS.push_back(curVet);
			tmpPositionWS.push_back(dataTruck->DTpositionWS[curi]);
			if (!dataTruck->DTnormalWS.empty())
				tmpNormalWS.push_back(dataTruck->DTnormalWS[curi]);
			if (!dataTruck->DTuv0.empty())
				tmpUV.push_back(dataTruck->DTuv0[curi]);
		}
	}
	dataTruck->DTpositionCS = tmpPositionCS;
	dataTruck->DTpositionWS = tmpPositionWS;
	dataTruck->DTnormalWS = tmpNormalWS;
	dataTruck->DTuv0 = tmpUV;
}

//齐次坐标裁剪
static void HomoClip(DataTruck* dataTruck)
{
	ClipWithPlane(ERROR, dataTruck);
	ClipWithPlane(RIGHT, dataTruck);
	ClipWithPlane(LEFT, dataTruck);
	ClipWithPlane(TOP, dataTruck);
	ClipWithPlane(BOTTOM, dataTruck);
	ClipWithPlane(NEAR, dataTruck);
	ClipWithPlane(FAR, dataTruck);
}

//RenderLoop拥有的dataTruck对象
DataTruck dataTruck;
/*
* RenderLoop负责将场景渲染到FrameBuffer上
* 包含颜色与深度信息
*/
static inline void RenderLoop(FrameBuffer* frameBuffer, FrameBuffer* shadowMap, Scene* mainScene, RenderConfig renderConfig)
{
	auto models = mainScene->GetModels();
	auto camera = (*mainScene->GetCameras())[0];//目前只有一个相机

	//获取当前shader的dataTruck
	dataTruck.camera = camera;
	dataTruck.WIDTH = WIDTH;
	dataTruck.HEIGHT = HEIGHT;
	dataTruck.shadowMap = shadowMap;
	for (int modelIdx = 0; modelIdx < models->size(); modelIdx++)//遍历所有模型
	{
		//获取当前模型
		auto model = (*models)[modelIdx];
		//更新MVP矩阵
		model->UpdateModelMatrix();
		camera->UpdateVPMatrix();
		dataTruck.matrixM = model->GetModelMatrix();
		dataTruck.matrixVP = camera->GetVPMatrix();
		dataTruck.model = model;
		Light mainLight = mainScene->GetLight();
		dataTruck.mainLight = mainLight;

		auto meshes = model->GetMeshes();
		//遍历每个模型的所有mesh
		for (int meshIdx = 0; meshIdx < meshes->size(); meshIdx++)
		{
			auto mesh = (*meshes)[meshIdx];
			auto pFace = mesh->GetPositionFaces();
			auto nFace = mesh->GetNormalFaces();
			auto vtFace = mesh->GetUVFaces();
			dataTruck.mesh = mesh;
			//根据config选取shader
			Shader* shader = nullptr;
			if (renderConfig == RENDER_SHADOW)
			{
				shader = mesh->GetShadowShader();
			}
			else if (renderConfig == RENDER_BY_PASS)
			{
				shader = mesh->GetCommonShader();
			}
			//将所选shader的dataTruck指向为renderLoop的dataTruck
			if (!shader)
			{
				continue;
			}
			shader->dataTruck = &dataTruck;
			//处理每个三角
			for (int i = 0; i < pFace->size(); i++)
			{
				dataTruck.Clear();

				Face positionFace = (*pFace)[i];
				Face normalFace = (*nFace)[i];
				Face uvFace = (*vtFace)[i];

				//加载顶点position
				if ((*mesh->GetPositions()).size() >= 3)
				{
					Eigen::Vector4f posA = (*mesh->GetPositions())[positionFace.A - 1];
					Eigen::Vector4f posB = (*mesh->GetPositions())[positionFace.B - 1];
					Eigen::Vector4f posC = (*mesh->GetPositions())[positionFace.C - 1];
					dataTruck.DTpositionOS.push_back(posA);
					dataTruck.DTpositionOS.push_back(posB);
					dataTruck.DTpositionOS.push_back(posC);
				}
				//加载顶点normal
				if ((*mesh->GetNormals()).size() >= 3)
				{
					dataTruck.DTnormalOS.push_back((*mesh->GetNormals())[normalFace.A - 1]);
					dataTruck.DTnormalOS.push_back((*mesh->GetNormals())[normalFace.B - 1]);
					dataTruck.DTnormalOS.push_back((*mesh->GetNormals())[normalFace.C - 1]);
				}
				//加载顶点uv
				if ((*mesh->GetTexcoords()).size() >= 3)
				{
					Eigen::Vector2f uvA = (*mesh->GetTexcoords())[uvFace.A - 1];
					Eigen::Vector2f uvB = (*mesh->GetTexcoords())[uvFace.B - 1];
					Eigen::Vector2f uvC = (*mesh->GetTexcoords())[uvFace.C - 1];
					dataTruck.DTuv0.push_back(uvA);
					dataTruck.DTuv0.push_back(uvB);
					dataTruck.DTuv0.push_back(uvC);
				}

				//运行顶点着色器
				shader->Vert();

				//背面剔除(Skybox不做)
				if (!model->IsSkyBox())
				{
					auto positionCS = dataTruck.DTpositionCS;
					Eigen::Vector3f v1 = (positionCS[1] / positionCS[1].w() - positionCS[0] / positionCS[0].w()).head(3);
					Eigen::Vector3f v2 = (positionCS[2] / positionCS[2].w() - positionCS[0] / positionCS[0].w()).head(3);
					Eigen::Vector3f vNormal = v1.cross(v2);
					if (vNormal.z() <= 0)
					{
						continue;
					}
				}

				//齐次坐标裁剪
				HomoClip(&dataTruck);
				int vertNum = dataTruck.DTpositionCS.size();
				dataTruck.DTpositionSS.resize(vertNum);
				for (int ti = 0; ti < vertNum - 2; ti++)
				{
					//获取三角包围盒
					auto positionCS = dataTruck.DTpositionCS;
					auto a = ComputeScreenPos(positionCS[0]);
					auto b = ComputeScreenPos(positionCS[ti + 1]);
					auto c = ComputeScreenPos(positionCS[ti + 2]);
					dataTruck.DTpositionSS[0] = a;
					dataTruck.DTpositionSS[ti + 1] = b;
					dataTruck.DTpositionSS[ti + 2] = c;
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
								float depth; 
								if (model->IsSkyBox())
								{
									depth = 1.f;
								}
								else
								{
									depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();
								}
								//深度测试
								if (depth > frameBuffer->GetZ(x, y))
								{
									continue;
								}

								//运行片元着色器
								auto finalColor = shader->Frag(Face(0, ti + 1, ti + 2), u.x(), u.y(), u.z());
								DrawPoint(frameBuffer, x, y, finalColor);
								frameBuffer->SetZ(x, y, depth);
							}
						}
					}
				}

			}
		}
	}
}