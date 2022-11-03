#pragma once
#include "imgui.h"
#include "Dense"
#include "FrameBuffer.h"
#include <iostream>

//常用颜色
const Eigen::Vector4f white = Eigen::Vector4f(255, 255, 255, 255);
const Eigen::Vector4f black = Eigen::Vector4f(0, 0, 0, 255);
const Eigen::Vector4f red = Eigen::Vector4f(255, 0, 0, 255);
const Eigen::Vector4f green = Eigen::Vector4f(0, 255, 0, 255);
const Eigen::Vector4f blue = Eigen::Vector4f(0, 0, 255, 255);

unsigned int Vector4fToColor(Eigen::Vector4f color)
{
	unsigned int rtnColor = 0;
	//小端CPU颜色编码
	rtnColor |= static_cast<uint8_t>(color.w()) << 24;
	rtnColor |= static_cast<uint8_t>(color.z()) << 16;
	rtnColor |= static_cast<uint8_t>(color.y()) << 8;
	rtnColor |= static_cast<uint8_t>(color.x()) << 0;
	return rtnColor;
}

Eigen::Vector4f normalColor(Eigen::Vector4f color)//将颜色分量截断在(0,255)
{
	return Eigen::Vector4f(std::min(255.f, std::max(0.f, color.x())), std::min(255.f, std::max(0.f, color.y())), std::min(255.f, std::max(0.f, color.z())), std::min(255.f, std::max(0.f, color.w())));
}


float normalX(FrameBuffer* frameBuffer, float x)
{
	x = std::max(x, 0.0f);
	x = std::min(x, (float)frameBuffer->width());
	return x;
}

float normalY(FrameBuffer* frameBuffer, float y)
{
	y = std::max(y, 0.0f);
	y = std::min(y, (float)frameBuffer->height());
	return y;
}

//在FrameBuffer的(x,y)位置画一个颜色为color的点,注意左上角坐标为(0,0)
void DrawPoint(FrameBuffer* frameBuffer, float x, float y, Eigen::Vector4f color = white)
{
	frameBuffer->SetColor(x, y, Vector4fToColor(normalColor(color)));
}


//从FrameBuffer的(x0,y0)到(x1,y1)画一条颜色为color的线,左上角坐标为(0,0)
void DrawLine(FrameBuffer* frameBuffer, float x0, float y0, float x1, float y1, Eigen::Vector4f color = white)
{
	if (x0 < 0 || y0 < 0 || x0 >= frameBuffer->width() || y0 >= frameBuffer->height())
		return;
	if (x1 < 0 || y1 < 0 || x1 >= frameBuffer->width() || y1 >= frameBuffer->height())
		return;
	//x0 = normalX(frameBuffer,x0);////todo FIX
	//x1 = normalX(frameBuffer,x1);
	//y0 = normalY(frameBuffer,y0);
	//y1 = normalY(frameBuffer,y1);

	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) {
		std::swap(x0, x1);
		std::swap(y0, y1);
	}
	int dx = x1 - x0;
	int dy = y1 - y0;
	int derror2 = std::abs(dy) * 2;
	int error2 = 0;
	int y = y0;
	for (int x = x0; x <= x1; x++) {
		if (steep) {
			DrawPoint(frameBuffer, y, x, color);
		}
		else {
			DrawPoint(frameBuffer, x, y, color);
		}
		error2 += derror2;
		if (error2 > dx) {
			y += (y1 > y0 ? 1 : -1);
			error2 -= dx * 2;
		}
	}
}

