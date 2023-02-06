#pragma once
#include "FrameBuffer.h"
#include "LowLevelAPI.h"

#include "DataPool.h"

//在FrameBuffer的(x,y)位置画一个颜色为color的点,左下角坐标为(0,0)
__device__ void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color);

__device__ void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth);


