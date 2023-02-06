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
//���ݴ���ṹ��
//����RenderLoop��Shader֮�����ݴ���

struct iblMap
{
	//��������ն���ͼ����GenerateIrradianceMapԤ������
	CubeMap irradianceMap;
	//Ԥ�˲�������ͼ
	int level;
	std::vector<CubeMap>* PrefilterMaps;
	//LUT
	Texture LUT;
};

struct DataTruck
{
	/*
	* ÿ֡����
	*/
	Eigen::Matrix4f matrixVP;
	Eigen::Vector3f lightDirTS;
	Eigen::Matrix4f lightMatrixVP;

	Light mainLight;
	Camera camera;
	FrameBuffer shadowMap;

	//������
	int WIDTH;
	int HEIGHT;
	iblMap iblMap;
	

	/*
	* ÿ��model����
	*/
	Eigen::Matrix4f matrixM;

	/*
	* ÿ��mesh����
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