#pragma once
#include "DataTruck.h"
#include "DataPool.h"
#include "Sampling.h"
#include "rasterize.cuh"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <queue>

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

	int size = clipVec2.size() - 2;
	for (int i = 0; i < size; i++)
	{
		ClipFragDatas.push_back(clipVec2[0]);
		ClipFragDatas.push_back(clipVec2[i + 1]);
		ClipFragDatas.push_back(clipVec2[i + 2]);
	}
}
void HomoClip()
{
	int size = ClipQueue.size();
	for (int i = 0; i < size - 2; i += 3)
	{
		HomoClipPreTrangle();
	}
}

void VertCal(Mesh* mesh, Eigen::Matrix4f matrixM, int shader)
{
	auto vertDatas = mesh->GetVertDatas();
	int size = vertDatas->size();
	FragDatas.resize(size);

	VertKernel(vertDatas, &FragDatas, &dataTruck, shader);
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
	ClipQueue.push(FragDatas[face.A]);
	ClipQueue.push(FragDatas[face.B]);
	ClipQueue.push(FragDatas[face.C]);
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

void CaculateLightMatrixVP(Light mainLight, Camera* camera)
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


static inline void RenderLoop(FrameBuffer* frameBuffer, FrameBuffer* shadowMap, Scene* mainScene, RenderConfig renderConfig)
{
	//单相机
	auto camera = (*mainScene->GetCameras())[0];
	Light mainLight = mainScene->GetLight();

	//将Global Data载入dataTruck
	{
		dataTruck.camera = *camera;
		dataTruck.matrixVP = camera->GetVPMatrix();
		dataTruck.mainLight = mainLight;
		dataTruck.WIDTH = WIDTH;
		dataTruck.HEIGHT = HEIGHT;
		dataTruck.shadowMap = *shadowMap;
	}
	//加载FrameBuffer数据到GPU内存
	LoadBufferData(BufferData(), BufferOffset());

	int currentShaderID = -1;
	camera->UpdateVPMatrix();

	if (renderConfig == RENDER_SHADOW)
	{
		camera->CalculateVisualCone();
		CaculateLightMatrixVP(mainLight, camera);
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
			currentShaderID = -1;
			auto mesh = (*meshes)[meshIdx];
			dataTruck.textures = mesh->GetTextures()->data();
			dataTruck.texNum = mesh->GetTextures()->size();
			if (mesh->m_CubeMap != nullptr)
			{
				dataTruck.cubeMap = *mesh->GetCubeMap();
			}

			if (renderConfig == RENDER_SHADOW)
			{
				currentShaderID = mesh->GetShadowShaderID();
			}
			else if (renderConfig == RENDER_BY_PASS)
			{
				currentShaderID = mesh->GetCommonShaderID();
			}
			if (currentShaderID == NONE)
			{
				goto EXIT;
			}

			//顶点着色
			VertCal(mesh, model->GetModelMatrix(), currentShaderID);
			//背面剔除
			CullBack(mesh, model->IsSkyBox());
			//齐次坐标裁剪
			HomoClip();
			//光栅化&像素着色
			FragKernel(*frameBuffer, &ClipFragDatas, &dataTruck, currentShaderID);
		}
	}
	//数据写回DataPool
	LoadBufferDeviceToHost();
EXIT:
	//释放GPU内存中FrameBuffer数据
	CudaFreeBufferData();
}
