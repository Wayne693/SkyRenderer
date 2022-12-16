#pragma once
#include "Dense"
#include "Model.h"

extern const int WIDTH;
extern const int HEIGHT;

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