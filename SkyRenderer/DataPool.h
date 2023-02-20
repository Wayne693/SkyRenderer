#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//high-level API
//处理纹理数据函数
 int AddTextureData(uint32_t* rawData, int size);

 const std::vector<uint32_t>* TexData();
 const std::vector<int>* Offset();

uint32_t GetTexData(int id, int pos);
void SetTexData(int id, int pos, uint32_t color);

__device__ uint32_t CudaGetTexData(int id, int pos);
__device__ void CudaSetTexData(int id, int pos, uint32_t color);

//处理FrameBuffer数据函数
 std::vector<uint32_t>* BufferData();

 __device__ float MinZAutomic(int id, int pos, float depth);

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
cudaError_t LoadTextureData(const std::vector<uint32_t>* rawData,const std::vector<int>* offset);
cudaError_t LoadBufferData(std::vector<uint32_t>* rawData, std::vector<int>* offset);
void CudaFreeBufferData();

//写回数据
cudaError_t LoadBufferDeviceToHost();