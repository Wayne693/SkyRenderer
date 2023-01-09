#pragma once
#include <vector>
#include "cuda_runtime.h"


//low-level API
__host__ __device__  uint32_t GetData(uint32_t* rawData, int* offset, int id, int pos);

__host__ __device__  void SetData(uint32_t* rawData, int* offset, int id, int pos, uint32_t color);

 int AddData(std::vector<uint32_t>& dstRawData, std::vector<int>& offset, uint32_t* srcRawData, int size);

 void initData(uint32_t* dstData, uint32_t val, int size);

//high-level API
//处理纹理数据函数
 int AddTextureData(uint32_t* rawData, int size);//

 std::vector<uint32_t>* RawData();//

 std::vector<int>* Offset();//

__host__ __device__  uint32_t GetRawData(int id, int pos);

__host__ __device__  void SetRawData(int id, int pos, uint32_t color);


//处理FrameBuffer数据函数
 std::vector<uint32_t>* BufferData();

 std::vector<int>* BufferOffset();

 int AddBufferData(uint32_t* rawData, int size);//

__host__ __device__  uint32_t GetBufferData(int id, int pos);

 uint32_t* GetBuffer(int id);

__host__ __device__  void SetBufferData(int id, int pos, uint32_t color);

void ClearBuffer(int id, int size, uint32_t color);

void ClearBuffer(int id, int size, float color);



