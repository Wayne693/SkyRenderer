#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Scene.h"
#include "Shader.h"
#include "Sampling.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>

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

//RenderLoop拥有的dataTruck对象
DataTruck dataTruck;



std::vector<Varyings> FragDatas;		//经顶点着色后的顶点数据
std::queue<Varyings> ClipQueue;			//裁剪后的顶点数据
std::vector<Varyings> ClipFragDatas;

std::mutex fbmutex;
std::thread th[4];


//三角重心插值，返回1-u-v,u,v
Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P)
{
	Eigen::Vector3f u = Eigen::Vector3f(B.x() - A.x(), C.x() - A.x(), A.x() - P.x()).cross(Eigen::Vector3f(B.y() - A.y(), C.y() - A.y(), A.y() - P.y()));// u v 1
	return Eigen::Vector3f(1.f - (u.x() + u.y()) / u.z(), u.x() / u.z(), u.y() / u.z());
}

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
static void ClipWithPlane(ClipPlane plane, std::vector<Varyings>& inVec, std::vector<Varyings>& outVec)
{
	outVec.clear();
	int size = inVec.size();
	for (int i = 0; i < size; i++)
	{
		int curi = i;
		int prei = (i - 1 + size) % size;
		Eigen::Vector4f curVet = inVec[curi].positionCS;
		Eigen::Vector4f preVet = inVec[prei].positionCS;

		int curin = IsInsidePlane(plane, curVet);
		int prein = IsInsidePlane(plane, preVet);
		//两顶点在平面两侧
		if (curin != prein)
		{
			float ratio = GetIntersectRatio(preVet, curVet, plane);
			Varyings tmpdata;
			tmpdata.positionCS = Lerp(preVet, curVet, ratio);
			tmpdata.positionWS = Lerp(inVec[prei].positionWS, inVec[curi].positionWS, ratio);
			tmpdata.normalWS = Lerp(inVec[prei].normalWS, inVec[curi].normalWS, ratio);
			tmpdata.tangentWS = Lerp(inVec[prei].tangentWS, inVec[curi].tangentWS, ratio);
			tmpdata.binormalWS = Lerp(inVec[prei].binormalWS, inVec[curi].binormalWS, ratio);
			tmpdata.uv = Lerp(inVec[prei].uv, inVec[curi].uv, ratio);
			outVec.push_back(tmpdata);
		}

		if (curin)
		{
			Varyings tmpdata;
			tmpdata.positionCS = curVet;
			tmpdata.positionWS = inVec[curi].positionWS;
			tmpdata.normalWS = inVec[curi].normalWS;
			tmpdata.tangentWS = inVec[curi].tangentWS;
			tmpdata.binormalWS = inVec[curi].binormalWS;
			tmpdata.uv = inVec[curi].uv;
			outVec.push_back(tmpdata);
		}
	}
}

//齐次坐标裁剪
static void HomoClipPreTrangle()
{
	std::vector<Varyings> clipVec1, clipVec2;
	for (int i = 0; i < 3; i++)
	{
		clipVec1.push_back(ClipQueue.front());
		ClipQueue.pop();
	}
	ClipWithPlane(ERROR, clipVec1, clipVec2);
	ClipWithPlane(RIGHT, clipVec2, clipVec1);
	ClipWithPlane(LEFT, clipVec1, clipVec2);
	ClipWithPlane(TOP, clipVec2, clipVec1);
	ClipWithPlane(BOTTOM, clipVec1, clipVec2);
	ClipWithPlane(NEAR, clipVec2, clipVec1);
	ClipWithPlane(FAR, clipVec1, clipVec2);

	//cqmutex.lock();
	int size = clipVec2.size() - 2;
	for (int i = 0; i < size; i++)
	{
		ClipFragDatas.push_back(clipVec2[0]);
		ClipFragDatas.push_back(clipVec2[i + 1]);
		ClipFragDatas.push_back(clipVec2[i + 2]);
	}
	//cqmutex.unlock();
}
void HomoClip()
{
	//ThreadPool threadpool(4);
	int size = ClipQueue.size();
	for (int i = 0; i < size - 2; i += 3)
	{
		//threadpool.Enqueue(HomoClipPreTrangle, i);
		HomoClipPreTrangle();
	}
}

void VertCal(Mesh* mesh, Eigen::Matrix4f matrixM, Shader* shader)
{
	
	shader->dataTruck = &dataTruck;
	auto vertDatas = mesh->GetVertDatas();
	int size = vertDatas->size();
	FragDatas.resize(size);

	const int blockNum = 4;
	int blocksize = size / blockNum + (size % blockNum != 0);
	for (int i = 0; i < blockNum; i++)
	{
		int begin = i * blocksize;
		int end = std::min(size, (i + 1) * blocksize);
		th[i] = std::thread([=] {
			for (int vertIdx = begin; vertIdx < end; vertIdx++)
			{
				Attributes tmpdata = { (*vertDatas)[vertIdx].positionOS,(*vertDatas)[vertIdx].normalOS,(*vertDatas)[vertIdx].tangentOS, (*vertDatas)[vertIdx].uv };
				FragDatas[vertIdx] = shader->Vert((*vertDatas)[vertIdx]);
			}
			});
	}
	for (int i = 0; i < blockNum; i++)
	{
		if (th[i].joinable())
		{
			th[i].join();
		}
	}
}


//背面剔除
bool ISBack(Eigen::Vector4f& posCSA, Eigen::Vector4f& posCSB, Eigen::Vector4f& posCSC)
{
	Eigen::Vector3f v1 = (posCSB / posCSB.w() - posCSA / posCSA.w()).head(3);
	Eigen::Vector3f v2 = (posCSC / posCSC.w() - posCSA / posCSA.w()).head(3);
	Eigen::Vector3f vNormal = v1.cross(v2);
	return vNormal.z() <= 0;
}
void CullFace(Face face, bool IsSkyBox)
{
	if (!IsSkyBox && ISBack(FragDatas[face.A].positionCS, FragDatas[face.B].positionCS, FragDatas[face.C].positionCS))
	{
		return;
	}
	//cqmutex.lock();
	ClipQueue.push(FragDatas[face.A]);
	ClipQueue.push(FragDatas[face.B]);
	ClipQueue.push(FragDatas[face.C]);
	//cqmutex.unlock();
}
void CullBack(Mesh* mesh, bool IsSkyBox)
{
	auto indexDatas = mesh->GetIndexDatas();
	for (int i = 0; i < indexDatas->size(); i++)
	{
		Face currentFace = (*indexDatas)[i];
		CullFace(currentFace, IsSkyBox);
	}
}


static inline void RenderLoop(FrameBuffer* frameBuffer, FrameBuffer* shadowMap, Scene* mainScene, RenderConfig renderConfig)
{
	auto camera = (*mainScene->GetCameras())[0];//目前只有一个相机

	//将Global Data载入dataTruck
	dataTruck.camera = camera;
	camera->UpdateVPMatrix();
	camera->CalculateVisualCone();
	dataTruck.matrixVP = camera->GetVPMatrix();
	Light mainLight = mainScene->GetLight();
	dataTruck.mainLight = mainLight;
	dataTruck.WIDTH = WIDTH;
	dataTruck.HEIGHT = HEIGHT;
	dataTruck.shadowMap = shadowMap;

	Shader* currentShader = nullptr;
	if (renderConfig == RENDER_SHADOW)
	{
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
		dataTruck.lightMatrixVP = matrixVP;
	}



	auto models = mainScene->GetModels();
	for (int modelIdx = 0; modelIdx < models->size(); modelIdx++)
	{
		auto model = (*models)[modelIdx];
		model->UpdateModelMatrix();
		auto meshes = model->GetMeshes();
		dataTruck.matrixM = model->GetModelMatrix();

		//遍历每个模型的所有mesh
		for (int meshIdx = 0; meshIdx < meshes->size(); meshIdx++)
		{
			FragDatas.clear();
			ClipFragDatas.clear();
			currentShader = nullptr;
			auto mesh = (*meshes)[meshIdx];
			dataTruck.mesh = mesh;

			if (renderConfig == RENDER_SHADOW)
			{
				currentShader = mesh->GetShadowShader();
			}
			else if (renderConfig == RENDER_BY_PASS)
			{
				currentShader = mesh->GetCommonShader();
			}
			//将所选shader的dataTruck指向为renderLoop的dataTruck
			if (!currentShader)
			{
				return;
			}

			
			//顶点着色
			VertCal(mesh, model->GetModelMatrix(), currentShader);

			//背面剔除
			CullBack(mesh, model->IsSkyBox());

			//齐次坐标裁剪
			HomoClip();


			int cnt = 0;
			const int blockNum = 4;
			clock_t c1 = clock();
			for (int i = 0; i < blockNum; i++)
			{
				int sizeM3 = ClipFragDatas.size() / 3;
				int blockSize = sizeM3 / blockNum + (sizeM3 % blockNum != 0);
				blockSize *= 3;
				th[i] = std::thread([=] {
					int begin = i * blockSize;
					int end = std::min((int)ClipFragDatas.size(), (i + 1) * blockSize);

					for (int vidx = begin; vidx < end; vidx += 3)
					{
						Varyings A, B, C;

						A = ClipFragDatas[vidx];
						B = ClipFragDatas[vidx + 1];
						C = ClipFragDatas[vidx + 2];

						auto a = ComputeScreenPos(A.positionCS);
						auto b = ComputeScreenPos(B.positionCS);
						auto c = ComputeScreenPos(C.positionCS);
						int minx = std::max(0, std::min(WIDTH, (int)std::min(a.x(), std::min(b.x(), c.x()))));
						int miny = std::max(0, std::min(HEIGHT, (int)std::min(a.y(), std::min(b.y(), c.y()))));
						int maxx = std::min(WIDTH, std::max(0, (int)std::max(a.x(), std::max(b.x(), c.x()))));
						int maxy = std::min(HEIGHT, std::max(0, (int)std::max(a.y(), std::max(b.y(), c.y()))));

						for (int x = minx; x <= maxx; x++)
						{
							for (int y = miny; y <= maxy; y++)
							{
								//三角插值
								Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));
								////像素在三角形内
								if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
								{
								//	//插值出深度
									float depth;
									if (!model->IsSkyBox())
									{
										depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();
									}
									else
									{
										depth = 1.0f;
									}

								//	//深度测试
									if (depth > frameBuffer->GetZ(x, y))
									{
										continue;
									}


									float alpha = u.x() / A.positionCS.w();
									float beta = u.y() / B.positionCS.w();
									float gamma = u.z() / C.positionCS.w();
									float zn = 1 / (alpha + beta + gamma);

									Varyings tmpdata;
									tmpdata.positionWS = zn * (alpha * A.positionWS + beta * B.positionWS + gamma * C.positionWS);
									if (!model->IsSkyBox())
									{
										tmpdata.positionCS = zn * (alpha * A.positionCS + beta * B.positionCS + gamma * C.positionCS);
										tmpdata.normalWS = zn * (alpha * A.normalWS + beta * B.normalWS + gamma * C.normalWS);
										tmpdata.tangentWS = zn * (alpha * A.tangentWS + beta * B.tangentWS + gamma * C.tangentWS);
										tmpdata.binormalWS = zn * (alpha * A.binormalWS + beta * B.binormalWS + gamma * C.binormalWS);
										tmpdata.uv = zn * (alpha * A.uv + beta * B.uv + gamma * C.uv);
									}
									
									
									//运行片元着色器 
									auto finalColor = currentShader->Frag(tmpdata);
									//auto finalColor = Eigen::Vector4f(0, 0, 0, 0);
									//fbmutex.lock();
									DrawPoint(frameBuffer, x, y, finalColor);
									frameBuffer->SetZ(x, y, depth);
									//fbmutex.unlock();
								}
							}
						}
					}

					});
			}

			for (int i = 0; i < blockNum; i++)
			{
				th[i].join();
			}
			clock_t c2 = clock();
			//printf("%lf\n", difftime(c2, c1));
		}
	}
}
