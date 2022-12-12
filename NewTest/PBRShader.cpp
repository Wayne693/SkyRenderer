#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

const float PI = acos(-1);
//Cook-Torrance BRDF

/*
*Trowbridge-Reita GGX 法线分布函数(NDF)
*DFG中D
*表示微平面中法线与h方向相同的比例
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
* Schlick-Beckmann GGX 几何函数
* DFG中G
* 微表面相互遮蔽
*/
//直接光照
float SchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float r = (1 + roughness);
	float k = r * r / 8.0;

	return nDotv / (nDotv * (1 - k) + k);
}
//IBL光照
float IBLSchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float k = roughness * roughness / 2.0;

	return nDotv / (nDotv * (1 - k) + k);
}
//史密斯法
//考虑观察方向遮蔽和光线方向遮蔽
float GeometrySmith(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f l, float roughness)
{
	float g1 = SchlickGGX(n, v, roughness);
	float g2 = SchlickGGX(n, l, roughness);

	return g1 * g2;
}

/*
* Fresnel
* 因观察角度与反射平面方向的夹角而引起反射程度不同
* DFG中F
* 可计算漫反射和镜面反射的比例
*/
//Fresnel Schlick近似
Eigen::Vector3f FresnelSchlick(Eigen::Vector3f h, Eigen::Vector3f v, Eigen::Vector3f F0)
{
	float hDotv = h.dot(v);

	return F0 + (Eigen::Vector3f(1.f, 1.f, 1.f) - F0) * pow(1 - hDotv, 5.f);
}
//IBL中Fresnel
Eigen::Vector3f FresnelSchlickRoughness(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f F0, float roughness)
{
	float nDotv = std::max(0.f, n.dot(v));
	float r1 = std::max(1.f - roughness, F0.x());
	float r2 = std::max(1.f - roughness, F0.y());
	float r3 = std::max(1.f - roughness, F0.z());
	return F0 + (Eigen::Vector3f(r1, r2, r3) - F0) * pow(1 - nDotv, 5.f);
}

Varyings PBRShader::Vert(Attributes vertex)
{
	auto matrixM = vertex.matrixM;
	auto matrixVP = dataTruck->matrixVP;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	Varyings o;

	//将positionOS转到positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = matrixVP * o.positionWS;

	//将normalOS转到normalWS
	Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
	o.normalWS = normalMatrix * vertex.normalOS;

	//将顶点uv坐标处理好
	//o.uv = TransformTex(vertex.uv, diabloDiffuse);
	return o;
}

Eigen::Vector4f PBRShader::Frag(Varyings i)
{
	////计算TBN
	//Eigen::Vector3f v1 = (dataTruck->DTpositionCS[face.B] / dataTruck->DTpositionCS[face.B].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	//Eigen::Vector3f v2 = (dataTruck->DTpositionCS[face.C] / dataTruck->DTpositionCS[face.C].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	//Eigen::Matrix3f A;
	//A << v1.x(), v1.y(), v1.z(),
	//	v2.x(), v2.y(), v2.z(),
	//	normalWS.x(), normalWS.y(), normalWS.z();
	//Eigen::Matrix3f AI = A.inverse();

	//Eigen::Vector3f i = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].x() - dataTruck->DTuv0[face.A].x(), dataTruck->DTuv0[face.C].x() - dataTruck->DTuv0[face.A].x(), 0);
	//Eigen::Vector3f j = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].y() - dataTruck->DTuv0[face.A].y(), dataTruck->DTuv0[face.C].y() - dataTruck->DTuv0[face.A].y(), 0);
	//i.normalize();
	//j.normalize();
	//Eigen::Matrix3f tbnMatrix;
	//tbnMatrix << i.x(), j.x(), normalWS.x(),
	//	i.y(), j.y(), normalWS.y(),
	//	i.z(), j.z(), normalWS.z();

	////std::cout << tbnMatrix << std::endl;

	////获取 texture
	//Texture* albedoTex = (*dataTruck->mesh->GetTextures())[0];
	//Texture* normalTex = (*dataTruck->mesh->GetTextures())[1];
	//Texture* roughnessTex = (*dataTruck->mesh->GetTextures())[2];
	//Texture* metallicTex = (*dataTruck->mesh->GetTextures())[3];
	//Texture* occlusionTex = (*dataTruck->mesh->GetTextures())[4];
	//Texture* emissionTex = (*dataTruck->mesh->GetTextures())[5];
	////采样
	////std::cout << uv << std::endl;
	//Eigen::Vector3f albedo = Tex2D(albedoTex, uv).head(3);
	//float roughness = Tex2D(roughnessTex, uv).x();
	//float metallic = Tex2D(metallicTex, uv).x();
	//float ao = Tex2D(occlusionTex, uv).x();
	//Eigen::Vector3f emission = Tex2D(emissionTex, uv).head(3);

	////获得法线纹理中法线数据
	//Eigen::Vector3f bumpTS = UnpackNormal(normalTex, uv);
	//Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();

	//Eigen::Vector3f worldPos = (a * dataTruck->DTpositionWS[face.A] + b * dataTruck->DTpositionWS[face.B] + c * dataTruck->DTpositionWS[face.C]).head(3);
	//Eigen::Vector3f viewDir = (dataTruck->camera->GetPosition() - worldPos).normalized();
	//
	////计算Fresnel项
	//Eigen::Vector3f F0(0.04f, 0.04f, 0.04f);
	//F0 = F0 + (albedo - F0) * metallic;
	//Eigen::Vector3f F = FresnelSchlickRoughness(normalWS, viewDir, F0, roughness);
	//Eigen::Vector3f Kd = Eigen::Vector3f(1.f, 1.f, 1.f) - F;

	//Eigen::Vector3f irradiance = dataTruck->iblMap.irradianceMap->GetData(normalWS).head(3);
	////diffuse
	//Eigen::Vector3f diffuse = Vec3Mul(Vec3Mul(Kd, irradiance), albedo);

	////specular
	//Eigen::Vector3f r = (2.f * viewDir.dot(normalWS) * normalWS - viewDir).normalized();
	//float nDotv = normalWS.dot(viewDir);
	//Eigen::Vector2f lutuv(nDotv, roughness);
	//Eigen::Vector3f lut = Tex2D(dataTruck->iblMap.LUT, lutuv).head(3);
	//Eigen::Vector3f specular = F0 * lut.x() + Eigen::Vector3f(lut.y(),lut.y(),lut.y());
	//int level = roughness * dataTruck->iblMap.level;
	//Eigen::Vector3f prefilter = (*(dataTruck->iblMap.PrefilterMaps))[level]->GetData(r).head(3);
	//specular = Vec3Mul(specular, prefilter);

	//Eigen::Vector3f fincol = (diffuse + specular) * ao + emission;

	//Eigen::Vector4f finalColor(fincol.x(), fincol.y(), fincol.z(), 1);
	//return finalColor;
	return Eigen::Vector4f(0, 0, 0, 0);
}