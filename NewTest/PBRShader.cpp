#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

const float PI = acos(-1);
//Cook-Torrance BRDF

/*
*Trowbridge-Reita GGX ���߷ֲ�����(NDF)
*DFG��D
*��ʾ΢ƽ���з�����h������ͬ�ı���
*/
float NDFGGXTR(Eigen::Vector3f n, Eigen::Vector3f h, float roughness)
{
	float nDoth = n.dot(h);
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float nDoth2 = nDoth * nDoth;
	float factor = nDoth2 * (alpha2 - 1) + 1;
	return alpha2 / (PI * factor * factor);
}

/*
* Schlick-Beckmann GGX ���κ���
* DFG��G
* ΢�����໥�ڱ�
*/
//ֱ�ӹ���
float SchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float r = (1 + roughness);
	float k = r * r / 8.0;

	return nDotv / (nDotv * (1 - k) + k);
}
//IBL����
float IBLSchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float k = roughness * roughness / 2.0;

	return nDotv / (nDotv * (1 - k) + k);
}
//ʷ��˹��
//���ǹ۲췽���ڱκ͹��߷����ڱ�
float GeometrySmith(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f l, float roughness)
{
	float g1 = SchlickGGX(n, v, roughness);
	float g2 = SchlickGGX(n, l, roughness);

	return g1 * g2;
}

/*
* Fresnel
* ��۲�Ƕ��뷴��ƽ�淽��ļнǶ�������̶Ȳ�ͬ
* DFG��F
* �ɼ���������;��淴��ı���
*/
//Fresnel Schlick����
Eigen::Vector3f FresnelSchlick(Eigen::Vector3f h, Eigen::Vector3f v, Eigen::Vector3f F0)
{
	float hDotv = h.dot(v);

	return F0 + (Eigen::Vector3f(1.f, 1.f, 1.f) - F0) * pow(1 - hDotv, 5.f);
}
//IBL��Fresnel
Eigen::Vector3f FresnelSchlickRoughness(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f F0, float roughness)
{
	float nDotv = std::max(0.f, n.dot(v));
	float r1 = std::max(1.f - roughness, F0.x());
	float r2 = std::max(1.f - roughness, F0.y());
	float r3 = std::max(1.f - roughness, F0.z());
	return F0 + (Eigen::Vector3f(r1, r2, r3) - F0) * pow(1 - nDotv, 5.f);
}

void PBRShader::Vert()
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

Eigen::Vector4f PBRShader::Frag(float a, float b, float c)
{
	//��ֵ����������(͸�ӽ�����ֵ)
	Eigen::Vector2f uv;
	float alpha = a / dataTruck->DTpositionWS[0].z();
	float beta = b / dataTruck->DTpositionWS[1].z();
	float gamma = c / dataTruck->DTpositionWS[2].z();
	float zn = 1 / (alpha + beta + gamma);
	uv = zn * (alpha * dataTruck->DTuv0[0] + beta * dataTruck->DTuv0[1] + gamma * dataTruck->DTuv0[2]);
	//��ֵ������
	Eigen::Vector3f normalWS = zn * (alpha * dataTruck->DTnormalWS[0] + beta * dataTruck->DTnormalWS[1] + gamma * dataTruck->DTnormalWS[2]);
	normalWS.normalize();
	//��ֵ����������(͸�ӽ�����ֵ)todo: understand
	Eigen::Vector4f positionWS = zn * (alpha * dataTruck->DTpositionWS[0] + beta * dataTruck->DTpositionWS[1] + gamma * dataTruck->DTpositionWS[2]);

	//����TBN
	Eigen::Vector3f v1 = (dataTruck->DTpositionCS[1] / dataTruck->DTpositionCS[1].w() - dataTruck->DTpositionCS[0] / dataTruck->DTpositionCS[0].w()).head(3);
	Eigen::Vector3f v2 = (dataTruck->DTpositionCS[2] / dataTruck->DTpositionCS[2].w() - dataTruck->DTpositionCS[0] / dataTruck->DTpositionCS[0].w()).head(3);
	Eigen::Matrix3f A;
	A << v1.x(), v1.y(), v1.z(),
		v2.x(), v2.y(), v2.z(),
		normalWS.x(), normalWS.y(), normalWS.z();
	Eigen::Matrix3f AI = A.inverse();

	Eigen::Vector3f i = AI * Eigen::Vector3f(dataTruck->DTuv0[1].x() - dataTruck->DTuv0[0].x(), dataTruck->DTuv0[2].x() - dataTruck->DTuv0[0].x(), 0);
	Eigen::Vector3f j = AI * Eigen::Vector3f(dataTruck->DTuv0[1].y() - dataTruck->DTuv0[0].y(), dataTruck->DTuv0[2].y() - dataTruck->DTuv0[0].y(), 0);
	i.normalize();
	j.normalize();
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << i.x(), j.x(), normalWS.x(),
		i.y(), j.y(), normalWS.y(),
		i.z(), j.z(), normalWS.z();

	//std::cout << tbnMatrix << std::endl;

	//��ȡ texture
	Texture* albedoTex = (*dataTruck->mesh->GetTextures())[0];
	Texture* normalTex = (*dataTruck->mesh->GetTextures())[1];
	Texture* roughnessTex = (*dataTruck->mesh->GetTextures())[2];
	Texture* metallicTex = (*dataTruck->mesh->GetTextures())[3];
	Texture* occlusionTex = (*dataTruck->mesh->GetTextures())[4];
	Texture* emissionTex = (*dataTruck->mesh->GetTextures())[5];
	//����
	//std::cout << uv << std::endl;
	Eigen::Vector3f albedo = Tex2D(albedoTex, uv).head(3);
	float roughness = Tex2D(roughnessTex, uv).x();
	float metallic = Tex2D(metallicTex, uv).x();
	float ao = Tex2D(occlusionTex, uv).x();
	Eigen::Vector3f emission = Tex2D(emissionTex, uv).head(3);

	//��÷��������з�������
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);
	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();

	Eigen::Vector3f worldPos = (a * dataTruck->DTpositionWS[0] + b * dataTruck->DTpositionWS[1] + c * dataTruck->DTpositionWS[2]).head(3);
	Eigen::Vector3f viewDir = (dataTruck->camera->GetPosition() - worldPos).normalized();
	
	//����Fresnel��
	Eigen::Vector3f F0(0.04f, 0.04f, 0.04f);
	F0 = F0 + (albedo - F0) * metallic;
	Eigen::Vector3f F = FresnelSchlickRoughness(normalWS, viewDir, F0, roughness);
	Eigen::Vector3f Kd = Eigen::Vector3f(1.f, 1.f, 1.f) - F;

	Eigen::Vector3f irradiance = dataTruck->iblMap.irradianceMap->GetData(normalWS).head(3);
	//diffuse
	Eigen::Vector3f diffuse = Vec3Mul(Vec3Mul(Kd, irradiance), albedo);

	//specular
	Eigen::Vector3f r = (2.f * viewDir.dot(normalWS) * normalWS - viewDir).normalized();
	float nDotv = normalWS.dot(viewDir);
	Eigen::Vector2f lutuv(nDotv, roughness);
	Eigen::Vector3f lut = Tex2D(dataTruck->iblMap.LUT, lutuv).head(3);
	Eigen::Vector3f specular = F0 * lut.x() + Eigen::Vector3f(lut.y(),lut.y(),lut.y());
	int level = roughness * dataTruck->iblMap.level;
	Eigen::Vector3f prefilter = (*(dataTruck->iblMap.PrefilterMaps))[level]->GetData(r).head(3);
	specular = Vec3Mul(specular, prefilter);

	Eigen::Vector3f fincol = (diffuse + specular) * ao + emission;

	Eigen::Vector4f finalColor(fincol.x(), fincol.y(), fincol.z(), 1);
	return finalColor;
}