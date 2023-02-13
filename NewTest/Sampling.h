#pragma once
#include "Dense"
#include "FrameBuffer.h"
#include "Model.h"
#include "cuda_runtime.h"

//常用颜色
const Eigen::Vector4f white = Eigen::Vector4f(255, 255, 255, 255);
const Eigen::Vector4f black = Eigen::Vector4f(0, 0, 0, 255);
const Eigen::Vector4f red = Eigen::Vector4f(255, 0, 0, 255);
const Eigen::Vector4f green = Eigen::Vector4f(0, 255, 0, 255);
const Eigen::Vector4f blue = Eigen::Vector4f(0, 0, 255, 255);

__host__ __device__ 
uint32_t Vector4fToColor(Eigen::Vector4f color);

__host__ __device__
Eigen::Vector4f ColorToVector4f(unsigned int color);

//将颜色分量截断在(0,255)
__device__
Eigen::Vector4f normalColor(Eigen::Vector4f color);

//颜色相乘
__host__ __device__
Eigen::Vector4f Vec4Mul(Eigen::Vector4f a, Eigen::Vector4f b);

__host__ __device__ 
Eigen::Vector3f Vec3Mul(Eigen::Vector3f a, Eigen::Vector3f b);

__host__ __device__
int selectCubeMapFace(Eigen::Vector3f direction, Eigen::Vector2f* texcoord);

//将uv坐标平移缩放
__host__ __device__ 
Eigen::Vector2f TransformTex(Eigen::Vector2f uv, Texture* texture);

//根据uv坐标采样纹理
__device__ 
Eigen::Vector4f Tex2D(Texture* texture, Eigen::Vector2f uv);

__device__ 
Eigen::Vector3f UnpackNormal(Texture* normalTexture, Eigen::Vector2f uv);

__device__ 
Eigen::Vector4f ComputeScreenPos(FrameBuffer* frameBuffer, Eigen::Vector4f positionCS);

//采样FrameBuffer的ZBuffer
__device__ 
float GetZ(FrameBuffer* buffer, int x, int y);

__device__ 
bool ZTestAutomic(FrameBuffer* buffer, float depth, int x, int y);

//采样CubeMap
__device__ 
Eigen::Vector4f CubeMapGetData(CubeMap* cubeMap, Eigen::Vector3f dir);

//在FrameBuffer的(x,y)位置画一个颜色为color的点,左下角坐标为(0,0)
__device__
void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color);

__device__
void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth);


