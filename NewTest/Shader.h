#pragma once

#include "Dense"
#include "vector"
#include "Scene.h"
#include "Model.h"

//���ݴ���ṹ��
//����RenderLoop��Shader֮�����ݴ���
//RenderLoop��Shaderֻ��Ҫ��עDataTruck�и��Ը���Ȥ������
//DataTruck������Ҳ��Ϊָ�룬�Ա�֤RenderLoop��Shader���Զ�дͬһƬ�ڴ�
struct DataTruck
{
	std::vector<Eigen::Vector4f> DTpositionOS;
	std::vector<Eigen::Vector3f> DTnormalOS;
	std::vector<Eigen::Vector2f> DTuv;
	std::vector<Eigen::Vector4f> DTpositionWS;
	std::vector<Eigen::Vector4f> DTpositionCS;
	std::vector<Eigen::Vector4f> DTpositionSS;
	std::vector<Eigen::Vector3f> DTnormalWS;

	Eigen::Matrix4f matrixM;
	Eigen::Matrix4f matrixVP;

	int WIDTH;
	int HEIGHT;

	Light mainLight;
	Model* model;
	Camera* camera;
};

class Shader
{
public:
	DataTruck dataTruck;
	virtual void Vert() = 0;
	virtual Eigen::Vector4f Frag(float a,float b,float c) = 0;//����Ϊ���ǲ�ֵ���1-u-v u v
	virtual void Clear() = 0;
};

class BlinnPhongShader :public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
	virtual void Clear();
};

static inline Eigen::Vector4f mulColor(Eigen::Vector4f a, Eigen::Vector4f b)//��ɫ���
{
	return Eigen::Vector4f(a.x() * b.x() / 255.0, a.y() * b.y() / 255.0, a.z() * b.z() / 255.0, a.w() * b.w() / 255.0);
}