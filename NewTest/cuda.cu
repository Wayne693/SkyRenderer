#include "cuda.cuh"
#include "Model.h"
#include "Dense"
#include "thrust/extrema.h"
//#include "device_atomic_functions.hpp"
//FrameBuffer cudaBuffer;

__device__ Eigen::Vector4f LambertFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer);

//三角重心插值，返回1-u-v,u,v
__host__ __device__ Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P)
{
	Eigen::Vector3f u = Eigen::Vector3f(B.x() - A.x(), C.x() - A.x(), A.x() - P.x()).cross(Eigen::Vector3f(B.y() - A.y(), C.y() - A.y(), A.y() - P.y()));// u v 1
	return Eigen::Vector3f(1.f - (u.x() + u.y()) / u.z(), u.x() / u.z(), u.y() / u.z());
}

__global__ void LambertVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= vertNum)
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

__global__ void PBRVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= vertNum)
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

__global__ void ShadowMapVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= vertNum)
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

__global__ void SkyBoxVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= vertNum)
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

//__global__ void test()
//{
//	printf("test ");
//}

//__device__ void testt(int x, int y)
//{
//	x = x + y;
//}

__device__ Eigen::Vector4f LambertFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer)
{
	auto mainLight = dataTruck->mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//计算TBN
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << i.tangentWS.x(), i.binormalWS.x(), i.normalWS.x(),
		i.tangentWS.y(), i.binormalWS.y(), i.normalWS.y(),
		i.tangentWS.z(), i.binormalWS.z(), i.normalWS.z();
	//获得法线纹理中法线数据
	Eigen::Vector3f bumpTS = UnpackNormal(&dataTruck->textures[1], i.uv);
	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();

	//diffuse
	float NdotL = bumpWS.dot(lightDirWS);
	Eigen::Vector4f diffuse = mainLight.intensity * thrust::max(NdotL, 0.f) * Vec4Mul(mainLight.color, Tex2D(&dataTruck->textures[0], i.uv));

	float shadow = 0.f;
	Eigen::Vector4f positionLSS = ComputeScreenPos(frameBuffer, dataTruck->lightMatrixVP * i.positionWS);
	float bias = thrust::max(0.05 * (1 - bumpWS.dot(lightDirWS)), 0.01);
	//PCF
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			shadow += (positionLSS.z() > GetZ(&dataTruck->shadowMap, positionLSS.x() + i, positionLSS.y() + j) + bias);
		}
	}
	shadow = thrust::min(0.7f, shadow / 9);

	Eigen::Vector4f finalColor = (1 - shadow) * diffuse;
	return finalColor;
}

__device__ Eigen::Vector4f ShadowMapFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer)
{
	float z = i.positionCS.z();
	z = (z + 1.f) / 2;
	Eigen::Vector4f depth(z, z, z, 1);
	Eigen::Vector4f finalColor = depth;
	return finalColor;
}

__global__ void CaculatePixel(FrameBuffer frameBuffer, Varyings* fragDatas, DataTruck* dataTruck, int shaderID,  Varyings* vertA, Varyings* vertB, Varyings* vertC, int x0, int y0, int w, int h)
{
	const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
	const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
	//printf("vertAadd = %p vertBadd = %p vertCadd = %p vertA(%lf, %lf, %lf) vertB(%lf, %lf, %lf) vertC(%lf, %lf, %lf)\n", &vertA, &vertB, &vertC, vertA->positionCS.x(), vertA->positionCS.y(), vertA->positionCS.z(), vertB->positionCS.x(), vertB->positionCS.y(), vertB->positionCS.z(), vertC->positionCS.x(), vertC->positionCS.y(), vertC->positionCS.z());

	auto a = ComputeScreenPos(&frameBuffer, vertA->positionCS);
	auto b = ComputeScreenPos(&frameBuffer, vertB->positionCS);
	auto c = ComputeScreenPos(&frameBuffer, vertC->positionCS);

	//printf("a(%lf, %lf, %lf) b(%lf, %lf, %lf)  c(%lf, %lf, %lf)\n", a.x(), a.y(), a.z(), b.x(), b.y(), b.z(), c.x(), c.y(), c.z());

	Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));

	if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
	{
		float depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();

		if (depth > GetZ(&frameBuffer, x, y))
		{
			return;
		}
		SetZ(&frameBuffer, x, y, depth);
		float alpha = u.x() / vertA->positionCS.w();
		float beta = u.y() / vertB->positionCS.w();
		float gamma = u.z() / vertC->positionCS.w();
		float zn = 1 / (alpha + beta + gamma);

		//插值
		Varyings tmpdata;
		tmpdata.positionWS = zn * (alpha * vertA->positionWS + beta * vertB->positionWS + gamma * vertC->positionWS);
		tmpdata.positionCS = zn * (alpha * vertA->positionCS + beta * vertB->positionCS + gamma * vertC->positionCS);
		tmpdata.normalWS = zn * (alpha * vertA->normalWS + beta * vertB->normalWS + gamma * vertC->normalWS);
		tmpdata.tangentWS = zn * (alpha * vertA->tangentWS + beta * vertB->tangentWS + gamma * vertC->tangentWS);
		tmpdata.binormalWS = zn * (alpha * vertA->binormalWS + beta * vertB->binormalWS + gamma * vertC->binormalWS);
		tmpdata.uv = zn * (alpha * vertA->uv + beta * vertB->uv + gamma * vertC->uv);

		Eigen::Vector4f finalColor = Eigen::Vector4f(0, 0, 0, 0);

		switch(shaderID)
		{
		case NONE:
			break;
		case LAMBERT_SHADER:
			finalColor = LambertFrag(tmpdata, dataTruck, &frameBuffer);
			break;
		case SHADOWMAP_SHADER:
			finalColor = ShadowMapFrag(tmpdata, dataTruck, &frameBuffer);
			break;
		};

		DrawPoint(&frameBuffer, x, y, finalColor);
		
	}
}

__global__ void CaculateTrangle(FrameBuffer frameBuffer, Varyings* fragDatas, DataTruck* dataTruck, int shaderID, int trangleNum, int offset)
{
	int idx = offset + (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= trangleNum)
	{
		return;
	}

	auto vertA = fragDatas[idx * 3];
	auto vertB = fragDatas[idx * 3 + 1];
	auto vertC = fragDatas[idx * 3 + 2];



	auto a = ComputeScreenPos(&frameBuffer, vertA.positionCS);
	auto b = ComputeScreenPos(&frameBuffer, vertB.positionCS);
	auto c = ComputeScreenPos(&frameBuffer, vertC.positionCS);
	// caculate AABB box
	int minx = thrust::max(0, thrust::min((int)frameBuffer.m_Width, (int)thrust::min(a.x(), thrust::min(b.x(), c.x()))));
	int miny = thrust::max(0, thrust::min((int)frameBuffer.m_Height, (int)thrust::min(a.y(), thrust::min(b.y(), c.y()))));
	int maxx = thrust::min((int)frameBuffer.m_Width, thrust::max(0, (int)thrust::max(a.x(), thrust::max(b.x(), c.x()))));
	int maxy = thrust::min((int)frameBuffer.m_Height, thrust::max(0, (int)thrust::max(a.y(), thrust::max(b.y(), c.y()))));
	//AABB包围盒的宽高
	int w = maxx - minx + 1;
	int h = maxy - miny + 1;

	const int threadNum = 8;

	dim3 blockNum(w / threadNum + (w % threadNum != 0), h / threadNum + (h % threadNum != 0));
	dim3 blockSize(threadNum, threadNum);

	CaculatePixel << <blockNum, blockSize >> > (frameBuffer, fragDatas, dataTruck, shaderID, fragDatas + idx * 3, fragDatas + idx * 3 + 1, fragDatas + idx * 3 + 2, minx, miny, w, h);
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("CaculatePixel Failed! : %s\n", cudaGetErrorString(cudaStatus));
	}


	//for (int x = minx; x <= maxx; x++)
	//{
	//	for (int y = miny; y <= maxy; y++)
	//	{
	//		//三角插值
	//		Eigen::Vector3f u = barycentric(Eigen::Vector2f(a.x(), a.y()), Eigen::Vector2f(b.x(), b.y()), Eigen::Vector2f(c.x(), c.y()), Eigen::Vector2f(x, y));
	//		//像素在三角形内
	//		if (u.x() >= 0 && u.y() >= 0 && u.z() >= 0)
	//		{
	//			//ii++;
	//			//插值出深度
	//			float depth;
	//			depth = u.x() * a.z() + u.y() * b.z() + u.z() * c.z();

	//			/*int writeindepth = depth;
	//			atomicMin(&writeindepth, (int)GetZ(&dataTruck->shadowMap, x, y));*/
	//			
	//			//Z-Test
	//			if (depth > GetZ(&frameBuffer, x, y))
	//			{
	//				continue;
	//			}
	//			SetZ(&frameBuffer, x, y, depth);

	//			float alpha = u.x() / vertA.positionCS.w();
	//			float beta = u.y() / vertB.positionCS.w();
	//			float gamma = u.z() / vertC.positionCS.w();
	//			float zn = 1 / (alpha + beta + gamma);

	//			//插值
	//			Varyings tmpdata;
	//			tmpdata.positionWS = zn * (alpha * vertA.positionWS + beta * vertB.positionWS + gamma * vertC.positionWS);
	//			tmpdata.positionCS = zn * (alpha * vertA.positionCS + beta * vertB.positionCS + gamma * vertC.positionCS);
	//			tmpdata.normalWS = zn * (alpha * vertA.normalWS + beta * vertB.normalWS + gamma * vertC.normalWS);
	//			tmpdata.tangentWS = zn * (alpha * vertA.tangentWS + beta * vertB.tangentWS + gamma * vertC.tangentWS);
	//			tmpdata.binormalWS = zn * (alpha * vertA.binormalWS + beta * vertB.binormalWS + gamma * vertC.binormalWS);
	//			tmpdata.uv = zn * (alpha * vertA.uv + beta * vertB.uv + gamma * vertC.uv);

	//			//运行片元着色器 
	//			auto finalColor = LambertFrag(tmpdata, dataTruck, &frameBuffer);
	//			//printf("color = %lf %lf %lf %lf\n", finalColor.x(), finalColor.y(), finalColor.z(), finalColor.w());
	//			//if (depth == GetZ(&frameBuffer, x, y))
	//			//{
	//				DrawPoint(&frameBuffer, x, y, finalColor);
	//			//}
	//			
	//			//printf("depth = %lf zBuffer = %lf\n", depth, GetZ(&frameBuffer, x, y));
	//		}
	//	}
	//}

}

/*
* 顶点着色
* 分配设备端内存&运行核函数&写回数据&释放设备端内存
*/
cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID)
{
	cudaError_t cudaStatus;

	Attributes* cudaVertDatas = nullptr;
	Varyings* cudaFragDatas = nullptr;
	DataTruck* cudaDataTruck = nullptr;
	//int* cudaVertNum = nullptr;
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

		////vertNum
		//cudaStatus = cudaMalloc((void**)&cudaVertNum, sizeof(int));
		//cudaStatus = cudaMemcpy(cudaVertNum, &vertNum, sizeof(int), cudaMemcpyHostToDevice);

		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess)
		//{
		//	printf("cudaFailed vertNum : %s", cudaGetErrorString(cudaStatus));
		//	goto Error;
		//}
	}

	//运行Kernel函数
	switch (shaderID)
	{
	case NONE:
		break;
	case LAMBERT_SHADER:
		LambertVert << <blockNum, threadNum >> > (cudaVertDatas, cudaFragDatas, cudaDataTruck, vertNum);
		break;
	case SHADOWMAP_SHADER:
		ShadowMapVert << <blockNum, threadNum >> > (cudaVertDatas, cudaFragDatas, cudaDataTruck, vertNum);
		break;
	case PBR_SHADER:
		PBRVert << <blockNum, threadNum >> > (cudaVertDatas, cudaFragDatas, cudaDataTruck, vertNum);
		break;
	case SKYBOX_SHADER:
		SkyBoxVert << <blockNum, threadNum >> > (cudaVertDatas, cudaFragDatas, cudaDataTruck, vertNum);
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
	//cudaFree(cudaVertNum);
	return cudaStatus;
}
/*
* 像素着色
* 分配设备端内存&运行核函数&写回数据&释放设备端内存
*/
cudaError_t FragKernel(FrameBuffer frameBuffer, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID)
{
	cudaError_t cudaStatus;

	Varyings* cudaFragDatas = nullptr;
	DataTruck* cudaDataTruck = nullptr;
	Texture* cudatmptextures = nullptr;
	auto tmpDataTruck = *dataTruck;
	
	const int threadNum = 192;
	const int kernelLimit = 192 * 7;
	int trangleNum = fragDatas->size() / 3;
	//int blockNum = trangleNum / threadNum + (trangleNum % threadNum != 0);
	int blockNum = kernelLimit / threadNum;
	int tnum = 0;


	//fragData
	cudaMalloc((void**)&cudaFragDatas, fragDatas->size() * sizeof(Varyings));
	cudaMemcpy(cudaFragDatas, fragDatas->data(), fragDatas->size() * sizeof(Varyings), cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFragDatas fail!");
		goto Error;
	}

	//dataTruck
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


	while (tnum < trangleNum)
	{
		CaculateTrangle << <blockNum, threadNum >> > (frameBuffer, cudaFragDatas, cudaDataTruck, shaderID, trangleNum, tnum);

		tnum += kernelLimit;

	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CaculateTrangle!\n", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(cudaFragDatas);
	cudaFree(cudatmptextures);
	cudaFree(cudaDataTruck);
	return cudaStatus;
}




//cudaError_t LoadFrameBuffer(FrameBuffer* frameBuffer)
//{
//	cudaError_t cudaStatus;
//
//	cudaMalloc((void**)&cudaBuffer, sizeof(FrameBuffer));
//	cudaMemcpy(cudaBuffer, frameBuffer, sizeof(FrameBuffer), cudaMemcpyHostToDevice);
//
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess)
//	{
//		printf("cudaDisplay Failed : %s", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//Error:
//	return cudaStatus;
//}
//
//void CudaFreeFrameBuffer()
//{
//	cudaFree(cudaBuffer);
//}
