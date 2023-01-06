#pragma once
#include <vector>
#include "cuda_runtime.h"

//纹理数据
static int textureNum = 0;
static std::vector<int> textureOffset;
static std::vector<uint32_t> textureRawData;

//FrameBuffer数据
static int bufferNum = 0;
static std::vector<int> bufferOffset;
static std::vector<uint32_t> bufferData;

//low-level API
static uint32_t GetData(uint32_t* rawData, int* offset, int id, int pos)
{
	return rawData[offset[id] + pos];
}

static void SetData(uint32_t* rawData,int* offset, int id, int pos, uint32_t color)
{
	rawData[offset[id] + pos] = color;
}

static int AddData(std::vector<uint32_t>& dstRawData, std::vector<int>& offset, uint32_t* srcRawData, int size)
{
	if (size <= 0)
	{
		return -1;
	}
	int datasize = dstRawData.size();
	dstRawData.resize(datasize + size);
	//若源数据为空指针则只分配空间不拷贝数据
	if (srcRawData != nullptr)
	{
		memcpy(dstRawData.data() + datasize, srcRawData, size * sizeof(uint32_t));
	}
	offset.push_back(datasize);
	return 0;
}

static void initData(uint32_t* dstData, uint32_t val, int size)
{
	std::fill_n(dstData, size, val);
}

//high-level API
//处理纹理数据函数
static int AddTextureData(uint32_t* rawData, int size)
{
	if (AddData(textureRawData, textureOffset, rawData, size) == 0)
	{
		return textureNum++;
	}
	printf("add texture fail\n");
	return -1;
}

static std::vector<uint32_t>* RawData()
{
	return &textureRawData;
}

static std::vector<int>* Offset()
{
	return &textureOffset;
}

static uint32_t GetRawData(int id, int pos)
{
	return GetData(textureRawData.data(), textureOffset.data(), id, pos);
}

static void SetRawData(int id, int pos, uint32_t color)
{
	SetData(textureRawData.data(), textureOffset.data(), id, pos, color);
}


//处理FrameBuffer数据函数
static std::vector<uint32_t>* BufferData()
{
	return &bufferData;
}

static std::vector<int>* BufferOffset()
{
	return &bufferOffset;
}

static int AddBufferData(uint32_t* rawData, int size)
{
	if (AddData(bufferData, bufferOffset, rawData, size) == 0)
	{
		return bufferNum++;
	}
	printf("add framebuffer fail\n");
	return -1;
}

static uint32_t GetBufferData(int id, int pos)
{
	return GetData(bufferData.data(), bufferOffset.data(), id, pos);
}

static uint32_t* GetBuffer(int id)
{
	return bufferData.data() + bufferOffset[id];
}

static void SetBufferData(int id, int pos, uint32_t color)
{
	SetData(bufferData.data(), bufferOffset.data(), id, pos, color);
}

template<typename T>
static void ClearBuffer(int id, int size, T color)
{
	initData(bufferData.data() + bufferOffset[id], *(uint32_t*)&color, size);
}

