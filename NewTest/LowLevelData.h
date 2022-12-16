#pragma once
#include "Dense"

struct Attributes
{
	Eigen::Vector4f positionOS;
	Eigen::Vector3f normalOS;
	Eigen::Vector4f tangentOS;//tangent(x,y,z) binormalsign(w)
	Eigen::Vector2f uv;
};

struct Varyings
{
	Eigen::Vector4f positionWS;
	Eigen::Vector4f positionCS;
	Eigen::Vector3f normalWS;
	Eigen::Vector3f tangentWS;//tangent(x,y,z)
	Eigen::Vector3f binormalWS;
	Eigen::Vector2f uv;
};

