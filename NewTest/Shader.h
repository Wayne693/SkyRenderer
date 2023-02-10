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
	//��������ն���ͼ����GenerateIrradianceMapԤ������
	CubeMap irradianceMap;
	//Ԥ�˲�������ͼ
	int level;
	std::vector<CubeMap>* PrefilterMaps;
	//LUT
	Texture LUT;
};

//���ݴ���ṹ��
//����RenderLoop��Shader֮�����ݴ���
struct DataTruck
{
	//ÿ֡����
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
	
	//ÿ��model����
	Eigen::Matrix4f matrixM;

	//ÿ��mesh����
	Texture* textures;
	int texNum;
	CubeMap cubeMap;
};
