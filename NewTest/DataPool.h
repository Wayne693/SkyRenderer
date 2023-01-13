#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

////low-level API
//__host__ __device__  uint32_t GetData(uint32_t* rawData, int* offset, int id, int pos);
//
//__host__ __device__  void SetData(uint32_t* rawData, int* offset, int id, int pos, uint32_t color);
//
// int AddData(std::vector<uint32_t>& dstRawData, std::vector<int>& offset, uint32_t* srcRawData, int size);
//
// void initData(uint32_t* dstData, uint32_t val, int size);

//high-level API
//处理纹理数据函数
 int AddTextureData(uint32_t* rawData, int size);//

 std::vector<uint32_t>* RawData();//

 std::vector<int>* Offset();//

uint32_t GetRawData(int id, int pos);
void SetRawData(int id, int pos, uint32_t color);

__device__ uint32_t CudaGetRawData(int id, int pos);
__device__ void CudaSetRawData(int id, int pos, uint32_t color);

//处理FrameBuffer数据函数
 std::vector<uint32_t>* BufferData();

 std::vector<int>* BufferOffset();

 int AddBufferData(uint32_t* rawData, int size);//

 uint32_t* GetBuffer(int id);

uint32_t GetBufferData(int id, int pos);
void SetBufferData(int id, int pos, uint32_t color);

__device__ uint32_t CudaGetBufferData(int id, int pos);
__device__ void CudaSetBufferData(int id, int pos, uint32_t color);

void ClearBuffer(int id, int size, uint32_t color);

void ClearBuffer(int id, int size, float color);

//将主存数据拷贝到显存
cudaError_t LoadTextureData(std::vector<uint32_t>* rawData, std::vector<int>* offset);
cudaError_t LoadBufferData(std::vector<uint32_t>* rawData, std::vector<int>* offset);
void CudaFreeBufferData();

//写回数据
cudaError_t LoadBufferDeviceToHost();