#include "Shader.h"
#include <iostream>

void NormalMapShader::Vert()
{
	auto matrixM = dataTruck.matrixM;
	auto matrixVP = dataTruck.matrixVP;
	int WIDTH = dataTruck.WIDTH;
	int HEIGHT = dataTruck.HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck.DTpositionWS.push_back(matrixM * dataTruck.DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck.DTpositionCS.push_back(matrixVP * dataTruck.DTpositionWS[i]);
		//将positionCS转到positionSS
		auto vertex = dataTruck.DTpositionCS[i];
		auto tmp = Eigen::Vector4f(vertex.x() * WIDTH / (2 * vertex.w()) + WIDTH / 2, vertex.y() * HEIGHT / (2 * vertex.w()) + HEIGHT / 2, vertex.z() / vertex.w(), vertex.w());
		dataTruck.DTpositionSS.push_back(tmp);

		//将normalOS转到normalWS
		auto normalos = dataTruck.DTnormalOS[i];
		Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
		Eigen::Vector3f normalWS = normalMatrix * normalos;
		dataTruck.DTnormalWS.push_back(normalWS);

		//将顶点uv坐标处理好
		TransformTex(&dataTruck.DTuv0, (*dataTruck.model->GetTextures())[0], i);
	}
}

Eigen::Vector4f NormalMapShader::Frag(float a, float b, float c)
{
	//插值出纹理坐标
	Eigen::Vector2f uv = a * dataTruck.DTuv0[0] + b * dataTruck.DTuv0[1] + c * dataTruck.DTuv0[2];
	//插值出法线
	Eigen::Vector3f normalWS = a * dataTruck.DTnormalWS[0] + b * dataTruck.DTnormalWS[1] + c * dataTruck.DTnormalWS[2];
	normalWS.normalize();

	float x = normalWS.x();
	float y = normalWS.y();
	float z = normalWS.z();
	Eigen::Vector3f tangentWS = Eigen::Vector3f(x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / std::sqrt(x * x + z * z));
	Eigen::Vector3f binormalWS = normalWS.cross(tangentWS);
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << tangentWS.x(), binormalWS.x(), normalWS.x(),
		tangentWS.y(), binormalWS.y(), normalWS.y(),
		tangentWS.z(), binormalWS.z(), normalWS.z();


	//获取diffuse texture、normal texture
	Texture* diffuseTex = (*dataTruck.model->GetTextures())[0];
	Texture* normalTex = (*dataTruck.model->GetTextures())[1];

	auto mainLight = dataTruck.mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//获得法线纹理中法线数据
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);
	bumpTS.head(2) *= 0.8f;
	bumpTS.z() = sqrt(1.f - std::max(0.f, std::min(1.f, bumpTS.head(2).dot(bumpTS.head(2)))));
	bumpTS.normalize();
	
	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();
	float NdotL = bumpWS.dot(lightDirWS);
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(mainLight.color, Tex2D(diffuseTex, uv));

	Eigen::Vector4f finalColor = diffuse;
	return finalColor;
}