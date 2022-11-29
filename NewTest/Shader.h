#pragma once

#include "Dense"
#include "vector"
#include "Scene.h"
#include "FrameBuffer.h"
#include <iostream>

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
	std::vector<Eigen::Vector4f> DTpositionOS;
	std::vector<Eigen::Vector3f> DTnormalOS;
	std::vector<Eigen::Vector2f> DTuv0;
	std::vector<Eigen::Vector4f> DTpositionWS;
	std::vector<Eigen::Vector4f> DTpositionCS;
	std::vector<Eigen::Vector4f> DTpositionSS;
	std::vector<Eigen::Vector3f> DTnormalWS;

	Eigen::Matrix4f matrixM;
	Eigen::Matrix4f matrixVP;
	Eigen::Vector3f lightDirTS;
	Eigen::Matrix4f lightMatrixVP;

	int WIDTH;
	int HEIGHT;

	Light mainLight;
	Model* model;
	Mesh* mesh;
	Camera* camera;
	FrameBuffer* shadowMap;
	iblMap iblMap;

	void Clear()
	{
		DTpositionOS.clear();
		DTpositionWS.clear();
		DTpositionCS.clear();
		DTpositionSS.clear();
		DTuv0.clear();
		DTnormalOS.clear();
		DTnormalWS.clear();
	}
};


class Shader
{
public:
	DataTruck* dataTruck;
	virtual void Vert() = 0;
	virtual Eigen::Vector4f Frag(Face face, float a,float b,float c) = 0;//参数为三角插值结果1-u-v u v s
};

class LambertShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(Face face, float a, float b, float c);
};

class NormalMapShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(Face face, float a, float b, float c);
};

class ShadowMapShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(Face face, float a, float b, float c);
};

class PBRShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(Face face,float a, float b, float c);
};

class SkyBoxShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(Face face, float a, float b, float c);
};

//颜色相乘
static inline Eigen::Vector4f Vec4Mul(Eigen::Vector4f a, Eigen::Vector4f b)
{
	return Eigen::Vector4f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

static inline Eigen::Vector3f Vec3Mul(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

//将uv坐标平移缩放
static inline void TransformTex(std::vector<Eigen::Vector2f>* uv, Texture* texture, int idx) 
{
	float x = (*uv)[idx].x() * texture->GetTilling().x() + texture->GetOffset().x();
	float y = (*uv)[idx].y() * texture->GetTilling().y() + texture->GetOffset().y();
	(*uv)[idx] = Eigen::Vector2f(x, y);
}
 
//根据uv坐标采样纹理
static inline Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv)
{
	return texture->GetData(uv);
}

static inline Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv)
{
	Eigen::Vector4f data = Tex2D(normalTexture, uv);
	return 2 * data.head(3) - Eigen::Vector3f(1, 1, 1);
}

static inline Eigen::Vector4f ComputeScreenPos(Eigen::Vector4f positionCS)
{
	return Eigen::Vector4f(positionCS.x() * WIDTH / (2 * positionCS.w()) + WIDTH / 2, positionCS.y() * HEIGHT / (2 * positionCS.w()) + HEIGHT / 2, positionCS.z() / positionCS.w(), positionCS.w());
}