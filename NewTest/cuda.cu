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
	//��positionWSת��positionCS
	o.positionCS = dataTruck->matrixVP * o.positionWS;
	//��normalOSת��normalWS
	Eigen::Matrix3f normalMatrix = dataTruck->matrixM.block(0, 0, 3, 3).transpose();
	Eigen::Matrix3f nn = normalMatrix.inverse();
	o.normalWS = nn * vertex.normalOS;
	//����tangentWS��binormalWS
	o.tangentWS = dataTruck->matrixM.block(0, 0, 3, 3) * vertex.tangentOS.head(3);
	o.binormalWS = o.normalWS.cross(o.tangentWS) * vertex.tangentOS.w();
	//������uv���괦���
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
	//��positionWSת��positionCS
	o.positionCS = dataTruck->matrixVP * o.positionWS;
	//��normalOSת��normalWS
	Eigen::Matrix3f normalMatrix = dataTruck->matrixM.block(0, 0, 3, 3).transpose();
	Eigen::Matrix3f nn = normalMatrix.inverse();
	o.normalWS = nn * vertex.normalOS;
	//����tangentWS��binormalWS
	o.tangentWS = dataTruck->matrixM.block(0, 0, 3, 3) * vertex.tangentOS.head(3);
	o.binormalWS = o.normalWS.cross(o.tangentWS) * vertex.tangentOS.w();
	//������uv���괦���
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
	//��positionOSת��positionWS
	o.positionWS = dataTruck->matrixM * vertex.positionOS;
	//��positionWSת��positionCS
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

	//��positionOSת��positionWS
	o.positionWS = dataTruck->matrixM * vertex.positionOS;
	//��positionWSת��positionCS
	o.positionCS = matrixP * matrixV * o.positionWS;
	fragDatas[idx] = o;
}

//������������ú˺������ͷ��ڴ�
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

	//�������ݵ�GPU�ڴ�
	{
		//��������
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

		//������ɫ������
		cudaStatus = cudaMalloc((void**)&cudaFragDatas, sizeof(Varyings) * fragDatas->size());
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed! cudaFragDatas");
			goto Error;
		}
		//dataTruck
		/*
		* �ṹ���к�ָ����������ڴ淽���ο�����
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

	//����Kernel����
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
	//ͬ��
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	//�������Device������Host
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

//������������(RenderLoopǰ����)
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

//����FrameBuffer����(ÿ֡����)
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