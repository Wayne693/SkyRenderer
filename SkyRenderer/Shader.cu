#include "Shader.cuh"
#include "Dense"
#include "thrust/extrema.h"
#include "PBR.cuh"
#include "Sampling.h"
#include "device_launch_parameters.h"

__device__ CubeMap* cudaPrefilterMaps = nullptr;
CubeMap* hostPrefilterMaps = nullptr;

#pragma region VertShader

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

#pragma endregion

#pragma region FragShader

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

__device__ Eigen::Vector4f PBRFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer)
{
	//获取 texture
	Texture* albedoTex = &dataTruck->textures[0];
	Texture* normalTex = &dataTruck->textures[1];
	Texture* roughnessTex = &dataTruck->textures[2];
	Texture* metallicTex = &dataTruck->textures[3];
	Texture* occlusionTex = &dataTruck->textures[4];
	Texture* emissionTex = &dataTruck->textures[5];
	//采样
	Eigen::Vector3f albedo = Tex2D(albedoTex, i.uv).head(3);
	float roughness = Tex2D(roughnessTex, i.uv).x();
	float metallic = Tex2D(metallicTex, i.uv).x();
	float ao = Tex2D(occlusionTex, i.uv).x();
	Eigen::Vector3f emission = Tex2D(emissionTex, i.uv).head(3);

	//计算TBN
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << i.tangentWS.x(), i.binormalWS.x(), i.normalWS.x(),
		i.tangentWS.y(), i.binormalWS.y(), i.normalWS.y(),
		i.tangentWS.z(), i.binormalWS.z(), i.normalWS.z();
	//获得法线纹理中法线数据
	Eigen::Vector3f bumpTS = UnpackNormal(normalTex, i.uv);
	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();

	Eigen::Vector3f worldPos = i.positionWS.head(3);
	Eigen::Vector3f viewDir = (dataTruck->camera.m_Position - worldPos).normalized();

	//计算Fresnel项
	Eigen::Vector3f F0(0.04f, 0.04f, 0.04f);
	F0 = F0 + (albedo - F0) * metallic;
	Eigen::Vector3f F = FresnelSchlickRoughness(bumpWS, viewDir, F0, roughness);
	Eigen::Vector3f Kd = Eigen::Vector3f(1.f, 1.f, 1.f) - F;

	Eigen::Vector3f irradiance = CubeMapGetData(&dataTruck->iblMap.irradianceMap, bumpWS).head(3);
	//diffuse
	Eigen::Vector3f diffuse = Vec3Mul(Vec3Mul(Kd, irradiance), albedo);

	//specular
	Eigen::Vector3f r = (2.f * viewDir.dot(bumpWS) * bumpWS - viewDir).normalized();
	float nDotv = bumpWS.dot(viewDir);
	Eigen::Vector2f lutuv(nDotv, roughness);
	Eigen::Vector3f lut = Tex2D(&dataTruck->iblMap.LUT, lutuv).head(3);
	Eigen::Vector3f specular = F0 * lut.x() + Eigen::Vector3f(lut.y(), lut.y(), lut.y());
	int level = roughness * dataTruck->iblMap.level;
	Eigen::Vector3f prefilter = CubeMapGetData(&cudaPrefilterMaps[level], r).head(3);
	specular = Vec3Mul(specular, prefilter);

	Eigen::Vector3f fincol = (diffuse + specular) * ao + emission;

	Eigen::Vector4f finalColor(fincol.x(), fincol.y(), fincol.z(), 1);
	return finalColor;
}

__device__ Eigen::Vector4f SkyBoxFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer)
{
	CubeMap cubeMap = dataTruck->cubeMap;
	//采样CubeMap
	Eigen::Vector4f finalColor = CubeMapGetData(&cubeMap, i.positionWS.head(3).normalized());
	return finalColor;
}

#pragma endregion


cudaError_t LoadPrefilterMaps(std::vector<CubeMap>* prefilterMaps)
{
	if (prefilterMaps == nullptr)////
		return cudaSuccess;

	cudaError_t cudaStatus;

	cudaMalloc((void**)&hostPrefilterMaps, prefilterMaps->size() * sizeof(CubeMap));
	cudaMemcpy(hostPrefilterMaps, prefilterMaps->data(), prefilterMaps->size() * sizeof(CubeMap), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cudaPrefilterMaps, &hostPrefilterMaps, sizeof(CubeMap*));

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("LoadPrefilterMapsDeviceToHostFailed : %s", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

