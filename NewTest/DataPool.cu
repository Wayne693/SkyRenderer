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
 uint32_t* cudaTexData;
 int* cudaTexOffset;
 uint32_t* cudaBufData;
 int* cudaBufOffset;

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
	return;
	//SetData(cudaBufData, cudaBufOffset, id, pos, color);
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

	cudaMalloc((void**)&cudaTexData, rawData->size() * sizeof(uint32_t));
	cudaMalloc((void**)&cudaTexOffset, offset->size() * sizeof(int));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(cudaTexData, rawData->data(), rawData->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaTexOffset, offset->data(), offset->size() * sizeof(int), cudaMemcpyHostToDevice);
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

	cudaMalloc((void**)&cudaBufData, rawData->size() * sizeof(uint32_t));
	cudaMalloc((void**)&cudaBufOffset, offset->size() * sizeof(int));

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaBufData Failed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(cudaBufData, rawData->data(), rawData->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaBufOffset, offset->data(), offset->size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaBufOffsetFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	return cudaStatus;
}

void CudaFreeBufferData()
{
	cudaFree(cudaBufData);
	cudaFree(cudaBufOffset);
}