#pragma once
#include "FrameBuffer.h"
#include "LowLevelAPI.h"

#include "DataPool.h"

//��FrameBuffer��(x,y)λ�û�һ����ɫΪcolor�ĵ�,���½�����Ϊ(0,0)
__device__ void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color);

__device__ void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth);


