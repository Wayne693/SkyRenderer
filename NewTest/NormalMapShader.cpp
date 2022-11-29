#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

void NormalMapShader::Vert()
{
	auto matrixM = dataTruck->matrixM;
	auto matrixVP = dataTruck->matrixVP;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//��positionOSת��positionWS
		dataTruck->DTpositionWS.push_back(matrixM * dataTruck->DTpositionOS[i]);
		//��positionWSת��positionCS
		dataTruck->DTpositionCS.push_back(matrixVP * dataTruck->DTpositionWS[i]);
		//��positionCSת��positionSS
		auto vertex = dataTruck->DTpositionCS[i];
		auto tmp = ComputeScreenPos(vertex);
		dataTruck->DTpositionSS.push_back(tmp);

		//��normalOSת��normalWS
		auto normalos = dataTruck->DTnormalOS[i];
		Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
		Eigen::Vector3f normalWS = normalMatrix * normalos;
		dataTruck->DTnormalWS.push_back(normalWS);

		//������uv���괦���
		TransformTex(&dataTruck->DTuv0, (*dataTruck->mesh->GetTextures())[0], i);
	}
}

Eigen::Vector4f NormalMapShader::Frag(Face face, float a, float b, float c)
{
	//��ֵ����������(͸�ӽ�����ֵ)
	Eigen::Vector2f uv;
	float alpha = a / dataTruck->DTpositionWS[face.A].z();
	float beta = b / dataTruck->DTpositionWS[face.B].z();
	float gamma = c / dataTruck->DTpositionWS[face.C].z();
	float zn = 1 / (alpha + beta + gamma);
	uv = zn * (alpha * dataTruck->DTuv0[face.A] + beta * dataTruck->DTuv0[face.B] + gamma * dataTruck->DTuv0[face.C]);
	//��ֵ������
	Eigen::Vector3f normalWS = a * dataTruck->DTnormalWS[face.A] + b * dataTruck->DTnormalWS[face.B] + c * dataTruck->DTnormalWS[face.C];
	normalWS.normalize();
	//��ֵ����������(͸�ӽ�����ֵ)todo: understand
	Eigen::Vector4f positionWS = zn * (alpha * dataTruck->DTpositionWS[face.A] + beta * dataTruck->DTpositionWS[face.B] + gamma * dataTruck->DTpositionWS[face.C]);

	//����TBN
	Eigen::Vector3f v1 = (dataTruck->DTpositionCS[face.B] / dataTruck->DTpositionCS[face.B].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	Eigen::Vector3f v2 = (dataTruck->DTpositionCS[face.C] / dataTruck->DTpositionCS[face.C].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	Eigen::Matrix3f A;
	A << v1.x(), v1.y(), v1.z(),
		v2.x(), v2.y(), v2.z(),
		normalWS.x(), normalWS.y(), normalWS.z();
	Eigen::Matrix3f AI = A.inverse();

	Eigen::Vector3f i = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].x() - dataTruck->DTuv0[face.A].x(), dataTruck->DTuv0[face.C].x() - dataTruck->DTuv0[face.A].x(), 0);
	Eigen::Vector3f j = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].y() - dataTruck->DTuv0[face.A].y(), dataTruck->DTuv0[face.C].y() - dataTruck->DTuv0[face.A].y(), 0);
	i.normalize();
	j.normalize();
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << i.x(), j.x(), normalWS.x(),
		i.y(), j.y(), normalWS.y(),
		i.z(), j.z(), normalWS.z();

	//std::cout << tbnMatrix << std::endl;
	
	//��ȡdiffuse texture��normal texture
	Texture* diffuseTex = (*dataTruck->mesh->GetTextures())[0];
	Texture* normalTex = (*dataTruck->mesh->GetTextures())[1];

	auto mainLight = dataTruck->mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//��÷��������з�������
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);


	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();
	float NdotL = bumpWS.dot(lightDirWS);
	//diffuse
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * Vec4Mul(mainLight.color, Tex2D(diffuseTex, uv));
	//specular
	Eigen::Vector3f worldPos = (a * dataTruck->DTpositionWS[face.A] + b * dataTruck->DTpositionWS[face.B] + c * dataTruck->DTpositionWS[face.C]).head(3);
	Eigen::Vector3f viewDir = (dataTruck->camera->GetPosition() - worldPos).normalized();
	Eigen::Vector3f halfDir = (viewDir + lightDirWS).normalized();
	Eigen::Vector4f specular = mainLight.intensity * mainLight.color * pow(std::max(0.f, bumpWS.dot(halfDir)), 25);

	float shadow = 0.f;
	if (GlobalSettings::GetInstance()->settings.drawShadow)
	{
		Eigen::Vector4f positionLSS = ComputeScreenPos(dataTruck->lightMatrixVP * positionWS);
		float bias = std::max(0.05 * (1 - bumpWS.dot(lightDirWS)), 0.01);
		//PCF
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				shadow += (positionLSS.z() > dataTruck->shadowMap->GetZ(positionLSS.x() + i, positionLSS.y() + j) + bias);
			}
		}
		shadow = std::min(0.7f, shadow / 9);
	}

	Eigen::Vector4f finalColor = (1 - shadow) * (diffuse + (shadow < 0.3) * specular);
	//Eigen::Vector4f finalColor(bumpWS.x() * 255, bumpWS.y() * 255, bumpWS.z() * 255, 255);
	//Eigen::Vector4f finalColor(bumpTS.x() * 255, bumpTS.y() * 255, bumpTS.z() * 255, 255);
	return finalColor;
}