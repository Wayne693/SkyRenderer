#pragma once
#include <iostream>
#include <Dense>
#include "LowLevelData.h"
#include "cuda_runtime.h"


//Cook-Torrance BRDF

/*
*Trowbridge-Reita GGX 法线分布函数(NDF)
*DFG中D
*表示微平面中法线与h方向相同的比例
*/
__device__
float NDFGGXTR(Eigen::Vector3f n, Eigen::Vector3f h, float roughness);


/*
* Schlick-Beckmann GGX 几何函数
* DFG中G
* 微表面相互遮蔽
*/
//直接光照
__device__
float SchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness);

//IBL光照
__device__
float IBLSchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness);
//史密斯法
//考虑观察方向遮蔽和光线方向遮蔽
__device__
float GeometrySmith(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f l, float roughness);

/*
* Fresnel
* 因观察角度与反射平面方向的夹角而引起反射程度不同
* DFG中F
* 可计算漫反射和镜面反射的比例
*/
//Fresnel Schlick近似
__device__
Eigen::Vector3f FresnelSchlick(Eigen::Vector3f h, Eigen::Vector3f v, Eigen::Vector3f F0);
//IBL中Fresnel
__device__
Eigen::Vector3f FresnelSchlickRoughness(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f F0, float roughness);


