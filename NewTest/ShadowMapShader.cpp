#include "Shader.h"
#include <iostream>


Varyings ShadowMapShader::Vert(Attributes vertex)
{
	auto matrixM = dataTruck->matrixM;
	auto mainLight = dataTruck->mainLight;
	auto camera = dataTruck->camera;

	Varyings o;

	//��positionOSת��positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//��positionWSת��positionCS
	o.positionCS = dataTruck->lightMatrixVP * o.positionWS;
	
	return o;
}

Eigen::Vector4f ShadowMapShader::Frag(Varyings i)
{
	float z = i.positionCS.z();
	z = (z + 1.f) / 2;
	Eigen::Vector4f depth(z, z, z, 1);
	Eigen::Vector4f finalColor = depth;
	return finalColor;
}