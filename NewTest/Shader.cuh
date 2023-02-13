#pragma once
#include "cuda_runtime.h"
#include "DataTruck.h"

__global__ void LambertVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum);

__global__ void PBRVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum);

__global__ void ShadowMapVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum);

__global__ void SkyBoxVert(Attributes* vertDatas, Varyings* fragDatas, DataTruck* dataTruck, int vertNum);

__device__ Eigen::Vector4f LambertFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer);

__device__ Eigen::Vector4f ShadowMapFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer);

__device__ Eigen::Vector4f PBRFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer);

__device__ Eigen::Vector4f SkyBoxFrag(Varyings i, DataTruck* dataTruck, FrameBuffer* frameBuffer);

//º”‘ÿ‘§¬À≤®ª∑æ≥Ã˘Õº
cudaError_t LoadPrefilterMaps(std::vector<CubeMap>* prefilterMaps);