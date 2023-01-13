#pragma once
#include "FrameBuffer.h"
#include "LowLevelAPI.h"

#include "DataPool.h"

//在FrameBuffer的(x,y)位置画一个颜色为color的点,左下角坐标为(0,0)
__device__ void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color);

__device__ void SetZ(FrameBuffer* frameBuffer, int x, int y, float depth);


////从FrameBuffer的(x0,y0)到(x1,y1)画一条颜色为color的线,左下角坐标为(0,0)
//static inline void DrawLine(FrameBuffer* frameBuffer, float x0, float y0, float x1, float y1, Eigen::Vector4f color = white)
//{
//	if (x0 < 0 || y0 < 0 || x0 >= frameBuffer->width() || y0 >= frameBuffer->height())
//		return;
//	if (x1 < 0 || y1 < 0 || x1 >= frameBuffer->width() || y1 >= frameBuffer->height())
//		return;
//
//	bool steep = false;
//	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
//		std::swap(x0, y0);
//		std::swap(x1, y1);
//		steep = true;
//	}
//	if (x0 > x1) {
//		std::swap(x0, x1);
//		std::swap(y0, y1);
//	}
//	int dx = x1 - x0;
//	int dy = y1 - y0;
//	int derror2 = std::abs(dy) * 2;
//	int error2 = 0;
//	int y = y0;
//	for (int x = x0; x <= x1; x++) {
//		if (steep) {
//			DrawPoint(frameBuffer, y, x, color);
//		}
//		else {
//			DrawPoint(frameBuffer, x, y, color);
//		}
//		error2 += derror2;
//		if (error2 > dx) {
//			y += (y1 > y0 ? 1 : -1);
//			error2 -= dx * 2;
//		}
//	}
//}

