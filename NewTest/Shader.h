#pragma once
#include "LowLevelAPI.h"
#include "LowLevelData.h"
#include "vector"
#include "Scene.h"
#include "FrameBuffer.h"
#include "Sampling.h"
#include <iostream>

#define NONE -1
#define LAMBERT_SHADER 0
#define PBR_SHADER 1
#define SHADOWMAP_SHADER 2
#define SKYBOX_SHADER 3

extern const int WIDTH;
extern const int HEIGHT;
//数据传输结构体
//用于RenderLoop与Shader之间数据传输

struct iblMap
{
	//漫反射辐照度贴图，由GenerateIrradianceMap预处理获得
	CubeMap* irradianceMap;
	//预滤波环境贴图
	int level;
	std::vector<CubeMap*>* PrefilterMaps;
	//LUT
	Texture* LUT;
};

struct DataTruck
{
	/*
	* 每帧更新
	*/
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
	
	/*
	* 每个model更新
	*/
	Eigen::Matrix4f matrixM;

	/*
	* 每个mesh更新
	*/
	Texture* textures;
	int texNum;
	CubeMap cubeMap;
};

class Shader
{
public:
	DataTruck* dataTruck;
	virtual Varyings Vert(Attributes vertex) = 0;
	virtual Eigen::Vector4f Frag(Varyings input) = 0;
};

class LambertShader : public Shader
{
public:
	virtual Varyings Vert(Attributes vertex);
	virtual Eigen::Vector4f Frag(Varyings input);
};

class NormalMapShader : public Shader
{
public:
	virtual Varyings Vert(Attributes vertex);
	virtual Eigen::Vector4f Frag(Varyings input);
};

class ShadowMapShader : public Shader
{
public:
	virtual Varyings Vert(Attributes vertex);
	virtual Eigen::Vector4f Frag(Varyings input);
};

class PBRShader : public Shader
{
public:
	virtual Varyings Vert(Attributes vertex);
	virtual Eigen::Vector4f Frag(Varyings input);
};

class SkyBoxShader : public Shader
{
public:
	virtual Varyings Vert(Attributes vertex);
	virtual Eigen::Vector4f Frag(Varyings input);
};