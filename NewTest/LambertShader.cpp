#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

void LambertShader::Vert()
{
	auto matrixM = dataTruck.matrixM;
	auto matrixVP = dataTruck.matrixVP;
	int WIDTH = dataTruck.WIDTH;
	int HEIGHT = dataTruck.HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//��positionOSת��positionWS
		dataTruck.DTpositionWS.push_back(matrixM * dataTruck.DTpositionOS[i]);
		//��positionWSת��positionCS
		dataTruck.DTpositionCS.push_back(matrixVP * dataTruck.DTpositionWS[i]);
		//��positionCSת��positionSS
		auto vertex = dataTruck.DTpositionCS[i];
		auto tmp = ComputeScreenSpace(vertex);
		dataTruck.DTpositionSS.push_back(tmp);

		//��normalOSת��normalWS
		auto normalos = dataTruck.DTnormalOS[i];
		Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
		Eigen::Vector3f normalWS = normalMatrix * normalos;
		dataTruck.DTnormalWS.push_back(normalWS);

		//������uv���괦���
		TransformTex(&dataTruck.DTuv0, (*dataTruck.model->GetTextures())[0], i);
	}
}

Eigen::Vector4f LambertShader::Frag(float a, float b, float c)
{
	//��ֵ����������(͸�ӽ�����ֵ)
	Eigen::Vector2f uv;
	float alpha = a / dataTruck.DTpositionWS[0].z();
	float beta = b / dataTruck.DTpositionWS[1].z();
	float gamma = c / dataTruck.DTpositionWS[2].z();
	float zn = 1 / (alpha + beta + gamma);
	uv = zn * (alpha * dataTruck.DTuv0[0] + beta * dataTruck.DTuv0[1] + gamma * dataTruck.DTuv0[2]);
	//��ֵ������
	Eigen::Vector3f normalWS = a * dataTruck.DTnormalWS[0] + b * dataTruck.DTnormalWS[1] + c * dataTruck.DTnormalWS[2];
	normalWS.normalize();
	//��ֵ����������
	Eigen::Vector4f positionWS = a * dataTruck.DTpositionWS[0] + b * dataTruck.DTpositionWS[1] + c * dataTruck.DTpositionWS[2];

	//����TBN
	float x = normalWS.x();
	float y = normalWS.y();
	float z = normalWS.z();
	Eigen::Vector3f tangentWS = Eigen::Vector3f(x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / std::sqrt(x * x + z * z));
	Eigen::Vector3f binormalWS = normalWS.cross(tangentWS);
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << tangentWS.x(), binormalWS.x(), normalWS.x(),
		tangentWS.y(), binormalWS.y(), normalWS.y(),
		tangentWS.z(), binormalWS.z(), normalWS.z();


	//��ȡdiffuse texture��normal texture
	Texture* diffuseTex = (*dataTruck.model->GetTextures())[0];
	Texture* normalTex = (*dataTruck.model->GetTextures())[1];

	auto mainLight = dataTruck.mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//��÷��������з�������
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);
	bumpTS.head(2) *= 0.8f;
	bumpTS.z() = sqrt(1.f - std::max(0.f, std::min(1.f, bumpTS.head(2).dot(bumpTS.head(2)))));
	bumpTS.normalize();

	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();
	float NdotL = bumpWS.dot(lightDirWS);
	//diffuse
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(mainLight.color, Tex2D(diffuseTex, uv));

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

	Eigen::Vector4f finalColor = (1 - shadow) * diffuse;
	return finalColor;
}