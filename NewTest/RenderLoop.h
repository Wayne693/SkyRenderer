#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Draw.h"
#include "Model.h"
#include "Camera.h"
#include "Scene.h"
#include "Shader.h"
#include "ThreadPool.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <time.h>

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

struct VertData
{
	Attributes rawData;
	Shader* shader;
};

struct FragData
{
	Varyings rawData;
	Shader* shader;
};

std::vector<FragData> FragDatas;		//经顶点着色后的顶点数据
std::queue<FragData> ClipQueue;		//裁剪后的顶点数据
std::vector<FragData> ClipFragDatas;
//std::vector<FragData> PixelDatas;
std::mutex fbmutex;

void foo()
{
	int a = 2;
	int b = 1;
	b = 1 + b + a;
	//printf("b = %d\n", b);
}


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
static void ClipWithPlane(ClipPlane plane, std::vector<FragData>& inVec, std::vector<FragData>& outVec)
{
	outVec.clear();
	int size = inVec.size();
	for (int i = 0; i < size; i++)
	{
		int curi = i;
		int prei = (i - 1 + size) % size;
		Eigen::Vector4f curVet = inVec[curi].rawData.positionCS;
		Eigen::Vector4f preVet = inVec[prei].rawData.positionCS;

		int curin = IsInsidePlane(plane, curVet);
		int prein = IsInsidePlane(plane, preVet);
		//两顶点在平面两侧
		if (curin != prein)
		{
			float ratio = GetIntersectRatio(preVet, curVet, plane);
			FragData tmpdata;
			tmpdata.shader = inVec[curi].shader;
			tmpdata.rawData.positionCS = Lerp(preVet, curVet, ratio);
			tmpdata.rawData.positionWS = Lerp(inVec[prei].rawData.positionWS, inVec[curi].rawData.positionWS, ratio);
			tmpdata.rawData.normalWS = Lerp(inVec[prei].rawData.normalWS, inVec[curi].rawData.normalWS, ratio);
			tmpdata.rawData.tangentWS = Lerp(inVec[prei].rawData.tangentWS, inVec[curi].rawData.tangentWS, ratio);
			tmpdata.rawData.binormalWS = Lerp(inVec[prei].rawData.binormalWS, inVec[curi].rawData.binormalWS, ratio);
			tmpdata.rawData.uv = Lerp(inVec[prei].rawData.uv, inVec[curi].rawData.uv, ratio);
			outVec.push_back(tmpdata);
		}

		if (curin)
		{
			FragData tmpdata;
			tmpdata.shader = inVec[curi].shader;
			tmpdata.rawData.positionCS = curVet;
			tmpdata.rawData.positionWS = inVec[curi].rawData.positionWS;
			tmpdata.rawData.normalWS = inVec[curi].rawData.normalWS;
			tmpdata.rawData.tangentWS = inVec[curi].rawData.tangentWS;
			tmpdata.rawData.binormalWS = inVec[curi].rawData.binormalWS;
			tmpdata.rawData.uv = inVec[curi].rawData.uv;
			outVec.push_back(tmpdata);
		}
	}
}

//齐次坐标裁剪
static void HomoClipPreTrangle()
{
	std::vector<FragData> clipVec1, clipVec2;
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


//顶点着色MultiThread
//void VertCal(Mesh* mesh, Eigen::Matrix4f matrixM, RenderConfig renderConfig)
//{
//	//ThreadPool threadpool(4);
//	//std::thread th(foo);
//	//th.join();
//	//const int workerNum = std::thread::hardware_concurrency();
//	//std::thread th[4];
//	Shader* shader = nullptr;
//	if (renderConfig == RENDER_SHADOW)
//	{
//		shader = mesh->GetShadowShader();
//	}
//	else if (renderConfig == RENDER_BY_PASS)
//	{
//		shader = mesh->GetCommonShader();
//	}
//	//将所选shader的dataTruck指向为renderLoop的dataTruck
//	if (!shader)
//	{
//		return;
//	}
//	shader->dataTruck = &dataTruck;
//	auto vertDatas = mesh->GetVertDatas();
//	int size = vertDatas->size();
//	FragDatas.resize(size);
//
//	const int blockNum = 4;
//	int blocksize = size / blockNum + (size % blockNum != 0);
//	for (int i = 0; i < blockNum; i++)
//	{
//		int begin = i * blocksize;
//		int end = std::min(size, (i + 1) * blocksize);
//		//th[i] = std::thread([=] {
//			for (int vertIdx = begin; vertIdx < end; vertIdx++)
//			{
//				VertData tmpdata = { {(*vertDatas)[vertIdx].positionOS,(*vertDatas)[vertIdx].normalOS,(*vertDatas)[vertIdx].tangentOS, (*vertDatas)[vertIdx].uv,matrixM},shader };
//				FragDatas[vertIdx] = { tmpdata.shader->Vert(tmpdata.rawData),shader };
//			}
//		//});
//	}
//	//for (int i = 0; i < blockNum; i++)
//	//{
//	//	if(th[i].joinable())
//	//	{
//	//		th[i].join();
//	//	}
//	//}
//}

void VertCal(Mesh* mesh, Eigen::Matrix4f matrixM, RenderConfig renderConfig)
{
	//ThreadPool threadpool(4);
	//std::thread th(foo);
	//th.join();
	//const int workerNum = std::thread::hardware_concurrency();
	std::thread th[4];
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
		return;
	}
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
				VertData tmpdata = { {(*vertDatas)[vertIdx].positionOS,(*vertDatas)[vertIdx].normalOS,(*vertDatas)[vertIdx].tangentOS, (*vertDatas)[vertIdx].uv,matrixM},shader };
				FragDatas[vertIdx] = { tmpdata.shader->Vert(tmpdata.rawData),shader };
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
	if (!IsSkyBox && ISBack(FragDatas[face.A].rawData.positionCS, FragDatas[face.B].rawData.positionCS, FragDatas[face.C].rawData.positionCS))
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
	//ThreadPool threadpool(4);
	auto indexDatas = mesh->GetIndexDatas();
	for (int i = 0; i < indexDatas->size(); i++)
	{
		Face currentFace = (*indexDatas)[i];
		//threadpool.Enqueue(CullFace, currentFace);
		CullFace(currentFace, IsSkyBox);
	}
}


static inline void RenderLoop(FrameBuffer* frameBuffer, FrameBuffer* shadowMap, Scene* mainScene, RenderConfig renderConfig)
{
	//printf("config = %d  ", renderConfig);

	auto camera = (*mainScene->GetCameras())[0];//目前只有一个相机
	//PixelDatas.clear();

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
		float minx, maxx, miny, maxy, minz, maxz = 0;
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

		//遍历每个模型的所有mesh

		for (int meshIdx = 0; meshIdx < meshes->size(); meshIdx++)
		{
			FragDatas.clear();
			ClipFragDatas.clear();
			//		//ThreadPool threadpool(4);
					//std::thread a[4];
			/*for (int i = 0; i < 1; i++)
			{
				a[i] = std::thread(foo);
			}
			for (int i = 0; i < 1; i++)
			{
				a[i].detach();
			}*/

			auto mesh = (*meshes)[meshIdx];
			dataTruck.mesh = mesh;///////////////todo
			//顶点着色

			VertCal(mesh, model->GetModelMatrix(), renderConfig);

			//背面剔除
			CullBack(mesh, model->IsSkyBox());

			//齐次坐标裁剪

			HomoClip();

			//ThreadPool threadpool(4);

			int cnt = 0;
			time_t t1 = clock();
			const int blockNum = 4;
			std::thread th[4];

			//printf("size = %d\n", ClipFragDatas.size());

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
						FragData A, B, C;

						A = ClipFragDatas[vidx];
						B = ClipFragDatas[vidx + 1];
						C = ClipFragDatas[vidx + 2];

						auto a = ComputeScreenPos(A.rawData.positionCS);
						auto b = ComputeScreenPos(B.rawData.positionCS);
						auto c = ComputeScreenPos(C.rawData.positionCS);
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
								//像素在三角形内
								if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
								{
									//插值出深度
									float depth;
									if (!model->IsSkyBox())
									{
										depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();
									}
									else
									{
										depth = 1.0f;
									}

									//深度测试
									if (depth > frameBuffer->GetZ(x, y))
									{
										continue;
									}


									float alpha = u.x() / A.rawData.positionCS.w();
									float beta = u.y() / B.rawData.positionCS.w();
									float gamma = u.z() / C.rawData.positionCS.w();
									float zn = 1 / (alpha + beta + gamma);

									FragData tmpdata;
									tmpdata.shader = A.shader;
									tmpdata.rawData.positionWS = zn * (alpha * A.rawData.positionWS + beta * B.rawData.positionWS + gamma * C.rawData.positionWS);
									tmpdata.rawData.positionCS = zn * (alpha * A.rawData.positionCS + beta * B.rawData.positionCS + gamma * C.rawData.positionCS);
									tmpdata.rawData.normalWS = zn * (alpha * A.rawData.normalWS + beta * B.rawData.normalWS + gamma * C.rawData.normalWS);
									tmpdata.rawData.tangentWS = zn * (alpha * A.rawData.tangentWS + beta * B.rawData.tangentWS + gamma * C.rawData.tangentWS);
									tmpdata.rawData.binormalWS = zn * (alpha * A.rawData.binormalWS + beta * B.rawData.binormalWS + gamma * C.rawData.binormalWS);
									tmpdata.rawData.uv = zn * (alpha * A.rawData.uv + beta * B.rawData.uv + gamma * C.rawData.uv);
									//PixelDatas.push_back(tmpdata);
									//运行片元着色器 

									auto finalColor = tmpdata.shader->Frag(tmpdata.rawData);
									fbmutex.lock();
									DrawPoint(frameBuffer, x, y, finalColor);
									frameBuffer->SetZ(x, y, depth);
									fbmutex.unlock();
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

			time_t t2 = clock();
			//printf("%lf\n", difftime(t2, t1));
		}
	}
}
