#include "cuda.cuh"
#include "Model.h"
#include "Dense"

uint32_t* cudaTexData;
int* cudaTexOffset;
uint32_t* cudaBufData;
int* cudaBufOffset;

__global__ void LambertVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int* vertNum)
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
	o.uv = TransformTex(vertex.uv, &dataTruck->textures[0]);
	fragDatas[idx] = o;
}

__global__ void PBRVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int* vertNum)
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
	o.uv = TransformTex(vertex.uv, &dataTruck->textures[0]);
	fragDatas[idx] = o;
}

__global__ void ShadowMapVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int* vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= *vertNum)
	{
		return;
	}
	Attributes vertex = vertDatas[idx];
	Varyings o;
	//将positionOS转到positionWS
	o.positionWS = dataTruck->matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = dataTruck->lightMatrixVP * o.positionWS;

	fragDatas[idx] = o;
}

__global__ void SkyBoxVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int* vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= *vertNum)
	{
		return;
	}
	Attributes vertex = vertDatas[idx];

	auto matrixP = dataTruck->camera.m_ProjectionMtx;
	Eigen::Matrix4f matrixV = Eigen::Matrix4f::Zero();
	matrixV << dataTruck->camera.m_ViewMtx.block(0, 0, 3, 3);
	matrixV(3, 3) = 1;

	Varyings o;

	//将positionOS转到positionWS
	o.positionWS = dataTruck->matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = matrixP * matrixV * o.positionWS;
	fragDatas[idx] = o;
}

//加载数据与调用核函数与释放内存
cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID)
{
	cudaError_t cudaStatus;


	Attributes* cudaVertDatas = nullptr;
	Varyings* cudaFragDatas = nullptr;
	DataTruck* cudaDataTruck = nullptr;
	int* cudaVertNum = nullptr;
	Texture* cudatmptextures = nullptr;

	const int threadNum = 192;
	int vertNum = vertDatas->size();
	int blockNum = vertDatas->size() / threadNum + (vertNum % threadNum != 0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//拷贝数据到GPU内存
	{
		//顶点数据
		cudaStatus = cudaMalloc((void**)&cudaVertDatas, sizeof(Attributes) * vertDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed! cudaVertDatas");
			goto Error;
		}
		cudaStatus = cudaMemcpy(cudaVertDatas, vertDatas->data(), vertNum * sizeof(Attributes), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed! cudaVertDatas");
			goto Error;
		}

		//顶点着色后数据
		cudaStatus = cudaMalloc((void**)&cudaFragDatas, sizeof(Varyings) * fragDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed! cudaFragDatas");
			goto Error;
		}
		//dataTruck
		/*
		* 结构体中含指针变量拷贝内存方法参考此文
		* https://devforum.nvidia.cn/forum.php?mod=viewthread&tid=6820&extra=&page=1
		*/
		
		cudaStatus = cudaMalloc((void**)&cudatmptextures, dataTruck->texNum * sizeof(Texture));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed! cudatmptextures");
			goto Error;
		}
		cudaStatus = cudaMemcpy(cudatmptextures, dataTruck->textures, dataTruck->texNum * sizeof(Texture), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemecpy failed! cudatmptextures");
			goto Error;
		}
		
		auto tmpDataTruck = *dataTruck;
		tmpDataTruck.textures = cudatmptextures;

		cudaStatus = cudaMalloc((void**)&cudaDataTruck, sizeof(DataTruck));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed! cudaDataTruck");
			goto Error;
		}
		cudaStatus = cudaMemcpy(cudaDataTruck, &tmpDataTruck, sizeof(DataTruck), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed! cudaDataTruck");
			goto Error;
		}

		//vertNum
		cudaStatus = cudaMalloc((void**)&cudaVertNum, sizeof(int));
		cudaStatus = cudaMemcpy(cudaVertNum, &vertNum, sizeof(int), cudaMemcpyHostToDevice);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaFailed vertNum : %s", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	}

	//运行Kernel函数
	switch (shaderID)
	{
	case NONE:
		break;
	case LAMBERT_SHADER:
		LambertVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaFragDatas, cudaDataTruck, cudaVertNum);
		break;
	case SHADOWMAP_SHADER:
		ShadowMapVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaFragDatas, cudaDataTruck, cudaVertNum);
		break;
	case PBR_SHADER:
		PBRVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaFragDatas, cudaDataTruck, cudaVertNum);
		break;
	case SKYBOX_SHADER:
		SkyBoxVert << <blockNum, threadNum >> > (cudaVertDatas, cudaFragDatas, cudaDataTruck, cudaVertNum);
		break;
	}
	

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
	cudaFree(cudatmptextures);
	cudaFree(cudaVertNum);
	return cudaStatus;
}

//加载纹理数据(RenderLoop前加载)
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

//加载FrameBuffer数据(每帧加载)
cudaError_t LoadBufferData(std::vector<uint32_t>* rawData, std::vector<int>* offset)
{
	cudaError_t cudaStatus;

	cudaMalloc((void**)&cudaBufData, rawData->size() * sizeof(uint32_t));
	cudaMalloc((void**)&cudaBufOffset, offset->size() * sizeof(int));
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(cudaBufData, rawData->data(), rawData->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaBufOffset, offset->data(), offset->size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFailed : %s", cudaGetErrorString(cudaStatus));
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