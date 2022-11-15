#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

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
		auto tmp = ComputeScreenSpace(vertex);
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
	//插值出纹理坐标(透视矫正插值)
	Eigen::Vector2f uv;
	float alpha = a / dataTruck.DTpositionWS[0].z();
	float beta = b / dataTruck.DTpositionWS[1].z();
	float gamma = c / dataTruck.DTpositionWS[2].z();
	float zn = 1 / (alpha + beta + gamma);
	uv = zn * (alpha * dataTruck.DTuv0[0] + beta * dataTruck.DTuv0[1] + gamma * dataTruck.DTuv0[2]);
	//插值出法线
	Eigen::Vector3f normalWS = a * dataTruck.DTnormalWS[0] + b * dataTruck.DTnormalWS[1] + c * dataTruck.DTnormalWS[2];
	normalWS.normalize();
	//插值出世界坐标
	Eigen::Vector4f positionWS = a * dataTruck.DTpositionWS[0] + b * dataTruck.DTpositionWS[1] + c * dataTruck.DTpositionWS[2];

	//计算TBN
	float x = normalWS.x();
	float y = normalWS.y();
	float z = normalWS.z();
	//printf("x = %lf y = %lf z = %lf\n", x, y, z);
	Eigen::Vector3f tangentWS = Eigen::Vector3f(x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / std::sqrt(x * x + z * z)).normalized();
	Eigen::Vector3f binormalWS = normalWS.cross(tangentWS).normalized();
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << tangentWS.x(), binormalWS.x(), normalWS.x(),
		tangentWS.y(), binormalWS.y(), normalWS.y(),
		tangentWS.z(), binormalWS.z(), normalWS.z();

	//std::cout << tbnMatrix << std::endl;

	//float deltaU1 = dataTruck.DTuv0[1].x() - dataTruck.DTuv0[0].x();
	//float deltaU2 = dataTruck.DTuv0[1].y() - dataTruck.DTuv0[0].y();
	//float deltaV1 = dataTruck.DTuv0[2].x() - dataTruck.DTuv0[0].x();
	//float deltaV2 = dataTruck.DTuv0[2].y() - dataTruck.DTuv0[0].y();
	//Eigen::Vector4f e1 = dataTruck.DTpositionWS[1] - dataTruck.DTpositionWS[0];
	//Eigen::Vector4f e2 = dataTruck.DTpositionWS[2] - dataTruck.DTpositionWS[0];

	//float ratio = 1 / (deltaU1 * deltaV2 - deltaU2 * deltaV1);
	//Eigen::Vector3f tangentWS(ratio * (deltaV2 * e1.x() - deltaV1 * e2.x()), ratio * (deltaV2 * e1.y() - deltaV1 * e2.y()), ratio * (deltaV2 * e1.z() - deltaV1 * e2.z()));
	//tangentWS.normalize();
	//Eigen::Vector3f binormalWS = normalWS.cross(tangentWS);
	//binormalWS.normalize();
	//Eigen::Matrix3f tbnMatrix;
	//tbnMatrix << tangentWS.x(), binormalWS.x(), normalWS.x(),
	//	tangentWS.y(), binormalWS.y(), normalWS.y(),
	//	tangentWS.z(), binormalWS.z(), normalWS.z();

	//获取diffuse texture、normal texture
	Texture* diffuseTex = (*dataTruck.model->GetTextures())[0];
	Texture* normalTex = (*dataTruck.model->GetTextures())[1];

	auto mainLight = dataTruck.mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//获得法线纹理中法线数据
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);
	//std::cout << bumpTS << std::endl;
	bumpTS.head(2) *= 0.8f;
	bumpTS.z() = sqrt(1.f - std::max(0.f, std::min(1.f, bumpTS.head(2).dot(bumpTS.head(2)))));
	bumpTS.normalize();

	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();
	float NdotL = bumpWS.dot(lightDirWS);
	//diffuse
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(mainLight.color, Tex2D(diffuseTex, uv));
	//specular
	Eigen::Vector3f worldPos = (a * dataTruck.DTpositionWS[0] + b * dataTruck.DTpositionWS[1] + c * dataTruck.DTpositionWS[2]).head(3);
	Eigen::Vector3f viewDir = (dataTruck.camera->GetPosition() - worldPos).normalized();
	Eigen::Vector3f halfDir = (viewDir + lightDirWS).normalized();
	Eigen::Vector4f specular = mainLight.intensity * mainLight.color * pow(std::max(0.f, bumpWS.dot(halfDir)), 25);

	float shadow = 0.f;
	if (GlobalSettings::GetInstance()->settings.drawShadow)
	{
		Eigen::Vector4f positionLSS = ComputeScreenSpace(dataTruck.lightMatrixVP * positionWS);
		float bias = std::max(0.05 * (1 - bumpWS.dot(lightDirWS)), 0.01);
		//PCF
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				shadow += (positionLSS.z() > dataTruck.shadowMap->GetZ(positionLSS.x() + i, positionLSS.y() + j) + bias);
			}
		}
		shadow = std::min(0.7f, shadow / 9);
	}

	Eigen::Vector4f finalColor = (1 - shadow) * (diffuse + (shadow < 0.3) * specular);
	//Eigen::Vector4f finalColor(bumpWS.x() * 255, bumpWS.y() * 255, bumpWS.z() * 255, 255);
	//Eigen::Vector4f finalColor(bumpTS.x() * 255, bumpTS.y() * 255, bumpTS.z() * 255, 255);
	return finalColor;
}