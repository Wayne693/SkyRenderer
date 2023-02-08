#include"DataPool.h"

/*
*主存中数据
* CPU端使用
*/
//纹理数据
 int textureNum;
 std::vector<int> textureOffset;
 std::vector<uint32_t> textureRawData;

//FrameBuffer数据
 int bufferNum;
 std::vector<int> bufferOffset;
 std::vector<uint32_t> bufferData;

 /*
 * 显存中数据
 * GPU端使用
 */
 __device__ uint32_t* cudaTexData;
 __device__ int* cudaTexOffset;
 __device__ uint32_t* cudaBufData;
 __device__ int* cudaBufOffset;
 //用于中转和释放device内存的变量
 uint32_t* cudaBufDataHost = nullptr;
 int* cudaBufOffsetHost = nullptr;

//low-level API
__host__ __device__  uint32_t GetData(uint32_t* rawData, int* offset, int id, int pos)
{
	return rawData[offset[id] + pos];
}

__host__ __device__  void SetData(uint32_t* rawData, int* offset, int id, int pos, uint32_t val)
{
	rawData[offset[id] + pos] = val;
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

 __device__ float fatomicMin(uint32_t* addr, float value)
 {
	 uint32_t old = *addr, assumed;
	 //printf("*up* addr = %lf value = %lf\n", __int_as_float(*addr), value);
	 //if (old <= value) return old;
	 do
	 {
		 assumed = old;
		 old = atomicCAS(addr, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
		 //printf("%")
	 } while (old != assumed);
	 //printf("*down* addr = %lf old = %lf\n", __int_as_float(*addr), __int_as_float(old));
	 return __int_as_float(*addr);
 }

//high-level API
//处理纹理数据函数
 int AddTextureData(uint32_t* rawData, int size)
{
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

uint32_t GetRawData(int id, int pos)
{
	return GetData(textureRawData.data(), textureOffset.data(), id, pos);
}

__device__ uint32_t CudaGetRawData(int id, int pos)
{
	return GetData(cudaTexData, cudaTexOffset, id, pos);
}

void SetRawData(int id, int pos, uint32_t color)
{
	SetData(textureRawData.data(), textureOffset.data(), id, pos, color);
}

__device__ void CudaSetRawData(int id, int pos, uint32_t color)
{
	SetData(cudaTexData, cudaTexOffset, id, pos, color);
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

uint32_t GetBufferData(int id, int pos)
{
	return GetData(bufferData.data(), bufferOffset.data(), id, pos);
}

__device__ uint32_t CudaGetBufferData(int id, int pos)
{
	return GetData(cudaBufData, cudaBufOffset, id, pos);
}

__device__ float MinZAutomic(int id, int pos, float depth)
{
	return fatomicMin(cudaBufData + cudaBufOffset[id] + pos, depth);
}

uint32_t* GetBuffer(int id)
{
	return bufferData.data() + bufferOffset[id];
}

void SetBufferData(int id, int pos, uint32_t color)
{
	SetData(bufferData.data(), bufferOffset.data(), id, pos, color);
}

__device__ void CudaSetBufferData(int id, int pos, uint32_t color)
{
	SetData(cudaBufData, cudaBufOffset, id, pos, color);
}

void ClearBuffer(int id, int size, uint32_t color) 
{
	initData(bufferData.data() + bufferOffset[id], color, size);
}

void ClearBuffer(int id, int size, float color)
{
	initData(bufferData.data() + bufferOffset[id], *(uint32_t*)&color, size);
}


cudaError_t LoadTextureData(std::vector<uint32_t>* rawData, std::vector<int>* offset)
{
	cudaError_t cudaStatus;

	//用于中转的临时变量
	uint32_t* cudaTexDataTmp = nullptr;
	int* cudaTexOffsetTmp = nullptr;

	//给临时变量在device端分配内存
	cudaMalloc((void**)&cudaTexDataTmp, rawData->size() * sizeof(uint32_t));
	cudaMalloc((void**)&cudaTexOffsetTmp, offset->size() * sizeof(int));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	//给临时变量在device端拷贝数据
	cudaMemcpy(cudaTexDataTmp, rawData->data(), rawData->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaTexOffsetTmp, offset->data(), offset->size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//将临时变量所存地址赋值给device变量
	cudaMemcpyToSymbol(cudaTexData, &cudaTexDataTmp, sizeof(uint32_t*));
	cudaMemcpyToSymbol(cudaTexOffset, &cudaTexOffsetTmp, sizeof(int*));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}
Error:
	return cudaStatus;
}


cudaError_t LoadBufferData(std::vector<uint32_t>* rawData, std::vector<int>* offset)
{
	cudaError_t cudaStatus;

	//给临时变量在device端分配内存
	cudaMalloc((void**)&cudaBufDataHost, rawData->size() * sizeof(uint32_t));
	cudaMalloc((void**)&cudaBufOffsetHost, offset->size() * sizeof(int));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaBufData Failed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//给临时变量在device端拷贝数据
	cudaMemcpy(cudaBufDataHost, rawData->data(), rawData->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaBufOffsetHost, offset->data(), offset->size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaBufOffsetFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//将临时变量所存地址赋值给device变量
	cudaMemcpyToSymbol(cudaBufData, &cudaBufDataHost, sizeof(uint32_t*));
	cudaMemcpyToSymbol(cudaBufOffset, &cudaBufOffsetHost, sizeof(int*));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t LoadBufferDeviceToHost()
{
	cudaMemcpy(bufferData.data(), cudaBufDataHost, bufferData.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(bufferOffset.data(), cudaBufOffsetHost, bufferOffset.size() * sizeof(int), cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("LoadBufferDataDeviceToHostFailed : %s", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}	

void CudaFreeBufferData()
{
	cudaFree(cudaBufDataHost);
	cudaFree(cudaBufOffsetHost);
}