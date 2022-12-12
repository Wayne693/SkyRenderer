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
	Eigen::Matrix4f matrixVP;
	Eigen::Vector3f lightDirTS;
	Eigen::Matrix4f lightMatrixVP;

	int WIDTH;
	int HEIGHT;

	Light mainLight;
	Camera* camera;
	FrameBuffer* shadowMap;
	iblMap iblMap;
};

struct Attributes
{
	Eigen::Vector4f positionOS;
	Eigen::Vector3f normalOS;
	Eigen::Vector2f uv;
	Eigen::Matrix4f matrixM;
};

struct Varyings
{
	Eigen::Vector4f positionWS;
	Eigen::Vector4f positionCS;
	Eigen::Vector3f normalWS;
	Eigen::Vector2f uv;
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
static inline Eigen::Vector2f TransformTex(Eigen::Vector2f uv, Texture* texture) 
{
	float x = uv.x() * texture->GetTilling().x() + texture->GetOffset().x();
	float y = uv.y() * texture->GetTilling().y() + texture->GetOffset().y();
	return Eigen::Vector2f(x, y);
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