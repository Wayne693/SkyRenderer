#pragma once
#include "Dense"
#include "FrameBuffer.h"
#include "Model.h"
#include "cuda_runtime.h"


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