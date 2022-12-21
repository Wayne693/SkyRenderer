#include "cuda.cuh"
#include "Model.h"
#include "Dense"

__global__ void CalVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int* vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx >= *vertNum)
	{
		return;
	}
	Attributes vertex = vertDatas[idx];
	Varyings o;
	o.positionWS = dataTruck->matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = dataTruck->matrixVP * o.positionWS;
	//将normalOS转到normalWS
	Eigen::Matrix3f normalMatrix = dataTruck->matrixM.block(0, 0, 3, 3).transpose();
	Eigen::Matrix3f nn = normalMatrix.inverse();
	o.normalWS = nn * vertex.normalOS;
	//计算tangentWS、binormalWS
	o.tangentWS = dataTruck->matrixM.block(0, 0, 3, 3) * vertex.tangentOS.head(3);
	o.binormalWS = o.normalWS.cross(o.tangentWS) * vertex.tangentOS.w();
	//将顶点uv坐标处理好
	float x = vertex.uv.x();
	float y = vertex.uv.y(); 
	o.uv = Eigen::Vector2f(x, y);

	fragDatas[idx] = o;
}


cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, Shader* shader)
{
	Attributes* cudaVertDatas = nullptr;
	Varyings* cudaFragDatas = nullptr;
	DataTruck* cudaDataTruck = nullptr;
	int* cudaVertNum = nullptr;
	//Texture** cudaTextures = nullptr;

	cudaError_t cudaStatus;
	const int threadNum = 192;
	int vertNum = vertDatas->size();
	int blockNum = vertDatas->size() / threadNum + (vertNum % threadNum != 0);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//给GPU分配内存
	{
		cudaStatus = cudaMalloc((void**)&cudaVertDatas, sizeof(Attributes) * vertDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&cudaFragDatas, sizeof(Varyings) * fragDatas->size());
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
		cudaMalloc((void**)&cudaVertNum, sizeof(int));
		cudaMemcpy(cudaVertNum, &vertNum, sizeof(int), cudaMemcpyHostToDevice);
		//auto textures = dataTruck->mesh->GetTextures();
		////以cudaTextures为首地址分配TextureList的内存
		//cudaStatus = cudaMalloc((void**)&cudaTextures, sizeof(Texture*) * textures->size());
		////printf("%d\n", sizeof(*cudaTextures));
		//for (int i = 0; i < textures->size(); i++)
		//{
		//	//Texture* currentTextureAdd = ;
		//	cudaStatus = cudaMalloc((void**)cudaTextures[i], sizeof(Texture));
		//	cudaMemcpy(cudaTextures[i], (*textures)[i], sizeof(Texture), cudaMemcpyHostToDevice);

		//	auto rawData = cudaTextures[i]->m_RawBuffer;
		//	cudaStatus = cudaMalloc((void**)&cudaTextures[i], sizeof((*textures)[i]->m_RawBuffer));
		//	cudaMemcpy(rawData, (*textures)[i]->m_RawBuffer, sizeof((*textures)[i]->m_RawBuffer), cudaMemcpyHostToDevice);
		//}

	}


	//数据拷贝
	{
		cudaStatus = cudaMemcpy(cudaVertDatas, vertDatas->data(), vertNum * sizeof(Attributes), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(cudaDataTruck, dataTruck, sizeof(DataTruck), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	//运行Kernel函数
	CalVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaFragDatas, cudaDataTruck,cudaVertNum);

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
	cudaStatus = cudaMemcpy(fragDatas->data(), cudaFragDatas, vertNum * sizeof(Varyings), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(cudaVertDatas);
	cudaFree(cudaFragDatas);
	cudaFree(cudaDataTruck);

	return cudaStatus;
}

