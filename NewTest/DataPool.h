#pragma once
#include <vector>
#include "cuda_runtime.h"

int textureNum = 0;
std::vector<int> textureOffset;
std::vector<uint32_t> textureRawData;

int AddTextureData(uint32_t* rawData, int size)
{
	if (size <= 0)
	{
		return -1;
	}
	int datasize = textureRawData.size();
	textureRawData.resize(datasize + size);
	memcpy(textureRawData.data() + datasize, rawData, size * sizeof(uint32_t));
	textureOffset.push_back(datasize);
	return textureNum++;
}

//__host__ __device__
uint32_t GetRawData(int id, int pos)
{
	return textureRawData[textureOffset[id] + pos];
}

void SetRawData(int id, int pos, uint32_t color)
{
	textureRawData[textureOffset[id] + pos] = color;
}