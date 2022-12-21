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

	//��GPU�����ڴ�
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
		////��cudaTexturesΪ�׵�ַ����TextureList���ڴ�
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


	//���ݿ���
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

	//����Kernel����
	CalVert <<<blockNum, threadNum >>> (cudaVertDatas, cudaFragDatas, cudaDataTruck,cudaVertNum);

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

	return cudaStatus;
}

