#include"DataPool.h"

//纹理数据
 int textureNum;
 std::vector<int> textureOffset;
 std::vector<uint32_t> textureRawData;

//FrameBuffer数据
 int bufferNum;
 std::vector<int> bufferOffset;
 std::vector<uint32_t> bufferData;

//low-level API
__host__ __device__  uint32_t GetData(uint32_t* rawData, int* offset, int id, int pos)
{
	return rawData[offset[id] + pos];
}

__host__ __device__  void SetData(uint32_t* rawData, int* offset, int id, int pos, uint32_t color)
{
	rawData[offset[id] + pos] = color;
}

 int AddData(std::vector<uint32_t>& dstRawData, std::vector<int>& offset, uint32_t* srcRawData, int size)
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

 void initData(uint32_t* dstData, uint32_t val, int size)
{
	std::fill_n(dstData, size, val);
}

//high-level API
//处理纹理数据函数
 int AddTextureData(uint32_t* rawData, int size)
{
	//printf("**size = %d", &textureRawData);/////////////////////////////////////test
	if (AddData(textureRawData, textureOffset, rawData, size) == 0)
	{
		return textureNum++;
	}
	printf("add texture fail\n");
	return -1;
}

 std::vector<uint32_t>* RawData()
{
	return &textureRawData;
}

 std::vector<int>* Offset()
{
	return &textureOffset;
}

__host__ __device__  uint32_t GetRawData(int id, int pos)
{
	//printf("texture data address = %d\n", &textureRawData);
	return GetData(textureRawData.data(), textureOffset.data(), id, pos);
}

__host__ __device__  void SetRawData(int id, int pos, uint32_t color)
{
	SetData(textureRawData.data(), textureOffset.data(), id, pos, color);
}


//处理FrameBuffer数据函数
std::vector<uint32_t>* BufferData()
{
	return &bufferData;
}

std::vector<int>* BufferOffset()
{
	return &bufferOffset;
}

int AddBufferData(uint32_t* rawData, int size)
{
	if (AddData(bufferData, bufferOffset, rawData, size) == 0)
	{
		return bufferNum++;
	}
	printf("add framebuffer fail\n");
	return -1;
}

__host__ __device__  uint32_t GetBufferData(int id, int pos)
{
	return GetData(bufferData.data(), bufferOffset.data(), id, pos);
}

 uint32_t* GetBuffer(int id)
{
	return bufferData.data() + bufferOffset[id];
}

__host__ __device__  void SetBufferData(int id, int pos, uint32_t color)
{
	SetData(bufferData.data(), bufferOffset.data(), id, pos, color);
}


void ClearBuffer(int id, int size, uint32_t color) 
{
	initData(bufferData.data() + bufferOffset[id], color, size);
}

void ClearBuffer(int id, int size, float color)
{
	initData(bufferData.data() + bufferOffset[id], *(uint32_t*)&color, size);
}
