#pragma once
#include <iostream>
#include <Dense>
#include "LowLevelData.h"
#include "cuda_runtime.h"


//Cook-Torrance BRDF

/*
*Trowbridge-Reita GGX ���߷ֲ�����(NDF)
*DFG��D
*��ʾ΢ƽ���з�����h������ͬ�ı���
*/
__device__
float NDFGGXTR(Eigen::Vector3f n, Eigen::Vector3f h, float roughness);


/*
* Schlick-Beckmann GGX ���κ���
* DFG��G
* ΢�����໥�ڱ�
*/
//ֱ�ӹ���
__device__
float SchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness);

//IBL����
__device__
float IBLSchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness);
//ʷ��˹��
//���ǹ۲췽���ڱκ͹��߷����ڱ�
__device__
float GeometrySmith(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f l, float roughness);

/*
* Fresnel
* ��۲�Ƕ��뷴��ƽ�淽��ļнǶ�������̶Ȳ�ͬ
* DFG��F
* �ɼ���������;��淴��ı���
*/
//Fresnel Schlick����
__device__
Eigen::Vector3f FresnelSchlick(Eigen::Vector3f h, Eigen::Vector3f v, Eigen::Vector3f F0);
//IBL��Fresnel
__device__
Eigen::Vector3f FresnelSchlickRoughness(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f F0, float roughness);


