#include "cuda.cuh"
#include "Model.h"

__global__ void CalVert(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, Varyings(*Vert) (Attributes))
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= vertDatas->size())
	{
		return;
	}
	(*fragDatas)[idx] = Vert((*vertDatas)[idx]);
}

cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Attributes>* fragDatas, DataTruck* dataTruck, Shader* shader)
{
	std::vector<Attributes>* cudaVertDatas = nullptr;
	//Shader* cudaShader = nullptr;
	std::vector<Attributes>* cudaFragDatas = nullptr;
	DataTruck* cudaDataTruck = nullptr;
	std::vector<Texture*>* cudaTextures = nullptr;

	cudaError_t cudaStatus;
	const int threadNum = 192;
	int blockNum = vertDatas->size() / threadNum + (cudaVertDatas->size() % threadNum);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//给GPU分配内存
	{
		cudaStatus = cudaMalloc((void**)&cudaVertDatas, sizeof(Attributes) * cudaVertDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		/*cudaStatus = cudaMalloc((void**)&cudaShader, sizeof(Shader));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}*/
		cudaStatus = cudaMalloc((void**)&cudaFragDatas, sizeof(Varyings) * cudaFragDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&cudaDataTruck, sizeof(DataTruck));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		auto textures = dataTruck->mesh->GetTextures();
		//以cudaTextures为首地址分配TextureList的内存
		cudaStatus = cudaMalloc((void**)&cudaTextures, sizeof(Texture*) * textures->size());

		for (int i = 0; i < textures->size(); i++)
		{
			auto currentTexture = (*cudaTextures)[i];
			cudaStatus = cudaMalloc((void**)&currentTexture, sizeof(Texture));
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
			auto rawData = currentTexture->m_RawBuffer;
			cudaStatus = cudaMalloc((void**)&rawData, sizeof((*textures)[i]->m_RawBuffer));
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
		}
		
	}


	//数据拷贝
	{
		cudaStatus = cudaMemcpy(cudaVertDatas, vertDatas, sizeof(Attributes) * cudaVertDatas->size(), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		/*cudaStatus = cudaMemcpy(cudaShader, shader, sizeof(Shader), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}*/
		cudaStatus = cudaMemcpy(cudaDataTruck, dataTruck, sizeof(DataTruck), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		auto textures = dataTruck->mesh->GetTextures();
		cudaStatus = cudaMalloc((void**)&cudaTextures, sizeof(Texture*) * textures->size());
		for (int i = 0; i < textures->size(); i++)
		{
			auto currentTexture = (*cudaTextures)[i];
			cudaStatus = cudaMalloc((void**)&currentTexture, sizeof(Texture));
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
			auto rawData = currentTexture->m_RawBuffer;
			cudaStatus = cudaMalloc((void**)&rawData, sizeof((*textures)[i]->m_RawBuffer));
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
		}
	}


	//运行Kernel函数
	CalVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaShader, cudaDataTruck, );

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//同步
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	//将结果从Device拷贝回Host
	cudaStatus = cudaMemcpy(fragDatas, cudaFragDatas, sizeof(Varyings) * cudaFragDatas->size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(cudaVertDatas);
	//cudaFree(cudaShader);
	cudaFree(cudaFragDatas);
	cudaFree(cudaDataTruck);
}