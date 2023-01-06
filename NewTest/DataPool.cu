//#include"DataPool.h"
//
////纹理数据
//int textureNum = 0;
//std::vector<int> textureOffset;
//std::vector<uint32_t> textureRawData;
//
////FrameBuffer数据
//int bufferNum = 0;
//std::vector<int> bufferOffset;
//std::vector<uint32_t> bufferData;
//
////low-level API
//uint32_t GetData(uint32_t* rawData, int id, int pos)
//{
//	return rawData[textureOffset[id] + pos];
//}
//
//void SetData(uint32_t* rawData, int id, int pos, uint32_t color)
//{
//	rawData[textureOffset[id] + pos] = color;
//}
//
//int AddData(std::vector<uint32_t>& dirRawData, std::vector<int>& offset, uint32_t* srcRawData, int size)
//{
//	if (size <= 0)
//	{
//		return -1;
//	}
//	int datasize = dirRawData.size();
//	dirRawData.resize(datasize + size);
//	//若源数据为空指针则只分配空间不拷贝数据
//	if (srcRawData != nullptr)
//	{
//		memcpy(dirRawData.data() + datasize, srcRawData, size * sizeof(uint32_t));
//	}
//	offset.push_back(datasize);
//	return 0;
//}
//
//void initData(uint32_t* dirData, uint32_t val, int size)
//{
//	std::fill_n(dirData, size, val);
//}
//
////处理纹理数据函数
//int AddTextureData(uint32_t* rawData, int size)
//{
//	if (AddData(textureRawData, textureOffset, rawData, size) == 0)
//	{
//		return textureNum++;
//	}
//	printf("add texture fail\n");
//	return -1;
//}
//
//std::vector<uint32_t>* RawData()
//{
//	return &textureRawData;
//}
//
//std::vector<int>* Offset()
//{
//	return &textureOffset;
//}
//
//uint32_t GetRawData(int id, int pos)
//{
//	return GetRawData(textureRawData.data(), id, pos);
//}
//
//void SetRawData(int id, int pos, uint32_t color)
//{
//	SetRawData(textureRawData.data(), id, pos, color);
//}
//
//
////处理FrameBuffer数据函数
//std::vector<uint32_t>* BufferData()
//{
//	return &bufferData;
//}
//
//std::vector<int>* BufferOffset()
//{
//	return &bufferOffset;
//}
//
//int AddBufferData(uint32_t* rawData, int size)
//{
//	if (AddData(bufferData, bufferOffset, rawData, size) == 0)
//	{
//		return bufferNum++;
//	}
//	printf("add framebuffer fail\n");
//	return -1;
//}
//
//uint32_t GetBufferData(int id, int pos)
//{
//	return GetRawData(bufferData.data(), id, pos);
//}
//
//void SetBufferData(int id, int pos, uint32_t color)
//{
//	SetRawData(bufferData.data(), id, pos, color);
//}