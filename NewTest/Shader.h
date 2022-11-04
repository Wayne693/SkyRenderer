#pragma once

#include "Dense"
#include "vector"
#include "Scene.h"
#include "Model.h"

//数据传输结构体
//用于RenderLoop与Shader之间数据传输
//RenderLoop和Shader只需要关注DataTruck中各自感兴趣的数据
//DataTruck中数据也多为指针，以保证RenderLoop和Shader可以读写同一片内存
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
	virtual Eigen::Vector4f Frag(float a,float b,float c) = 0;//参数为三角插值结果1-u-v u v
	virtual void Clear() = 0;
};

class BlinnPhongShader :public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
	virtual void Clear();
};

static inline Eigen::Vector4f mulColor(Eigen::Vector4f a, Eigen::Vector4f b)//颜色相乘
{
	return Eigen::Vector4f(a.x() * b.x() / 255.0, a.y() * b.y() / 255.0, a.z() * b.z() / 255.0, a.w() * b.w() / 255.0);
}