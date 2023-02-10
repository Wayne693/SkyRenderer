#pragma once
#include "LowLevelData.h"
#include "vector"
#include "FrameBuffer.h"
#include "Scene.h"


#define NONE -1
#define LAMBERT_SHADER 0
#define PBR_SHADER 1
#define SHADOWMAP_SHADER 2
#define SKYBOX_SHADER 3

struct iblMap
{
	//漫反射辐照度贴图，由GenerateIrradianceMap预处理获得
	CubeMap irradianceMap;
	//预滤波环境贴图
	int level;
	std::vector<CubeMap>* PrefilterMaps;
	//LUT
	Texture LUT;
};

//数据传输结构体
//用于RenderLoop与Shader之间数据传输
struct DataTruck
{
	//每帧更新
	Eigen::Matrix4f matrixVP;
	Eigen::Vector3f lightDirTS;
	Eigen::Matrix4f lightMatrixVP;

	Light mainLight;
	Camera camera;
	FrameBuffer shadowMap;

	//不更新
	int WIDTH;
	int HEIGHT;
	iblMap iblMap;
	
	//每个model更新
	Eigen::Matrix4f matrixM;

	//每个mesh更新
	Texture* textures;
	int texNum;
	CubeMap cubeMap;
};
