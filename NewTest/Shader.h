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
struct DataTruck
{
	std::vector<Eigen::Vector4f> DTpositionOS;
	std::vector<Eigen::Vector3f> DTnormalOS;
	std::vector<Eigen::Vector3f> DTtangentOS;
	std::vector<Eigen::Vector2f> DTuv0;
	std::vector<Eigen::Vector2f> DTuv1;
	std::vector<Eigen::Vector4f> DTpositionWS;
	std::vector<Eigen::Vector3f> DTtangentWS;
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

	void Clear()
	{
		DTpositionOS.clear();
		DTpositionWS.clear();
		DTpositionCS.clear();
		DTpositionSS.clear();
		DTuv0.clear();
		DTuv1.clear();
		DTnormalOS.clear();
		DTnormalWS.clear();
		DTtangentOS.clear();
		DTtangentWS.clear();
	}
};


class Shader
{
public:
	DataTruck* dataTruck;
	virtual void Vert() = 0;
	virtual Eigen::Vector4f Frag(float a,float b,float c) = 0;//参数为三角插值结果1-u-v u v s
};

class LambertShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

class BlinnPhongShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

class NormalMapShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

class ShadowMapShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

class PBRShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

class SkyBoxShader : public Shader
{
public:
	virtual void Vert();
	virtual Eigen::Vector4f Frag(float a, float b, float c);
};

//颜色相乘
static inline Eigen::Vector4f mulColor(Eigen::Vector4f a, Eigen::Vector4f b)
{
	return Eigen::Vector4f(a.x() * b.x() / 255.0, a.y() * b.y() / 255.0, a.z() * b.z() / 255.0, a.w() * b.w() / 255.0);
}
//将uv坐标平移缩放
static inline void TransformTex(std::vector<Eigen::Vector2f>* uv, Texture* texture, int idx) 
{
	float x = (*uv)[idx].x() * texture->GetTilling().x() * texture->width() + texture->GetOffset().x();
	float y = (*uv)[idx].y() * texture->GetTilling().y() * texture->height() + texture->GetOffset().y();
	(*uv)[idx] = Eigen::Vector2f(x, y);
}

static inline void TransformTex(Eigen::Vector2f* uv, Texture* texture)
{
	//std::cout << texture->width() << " " << texture->height() << std::endl;
	float x = uv->x() * texture->GetTilling().x() + texture->GetOffset().x();
	float y = uv->y() * texture->GetTilling().y() + texture->GetOffset().y();
	(*uv) = Eigen::Vector2f(x, y);
}
//根据uv坐标采样纹理
static inline Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv)
{
	return texture->GetData(uv);
}

static inline Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv)
{
	Eigen::Vector4f data = Tex2D(normalTexture, uv) / 255.f;
	return 2 * data.head(3) - Eigen::Vector3f(1, 1, 1);
}

static inline Eigen::Vector4f ComputeScreenPos(Eigen::Vector4f positionCS)
{
	return Eigen::Vector4f(positionCS.x() * WIDTH / (2 * positionCS.w()) + WIDTH / 2, positionCS.y() * HEIGHT / (2 * positionCS.w()) + HEIGHT / 2, positionCS.z() / positionCS.w(), positionCS.w());
}