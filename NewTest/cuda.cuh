#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Shader.h"
#include "DataPool.h";
#include <vector>
#include <stdio.h>


cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID);

cudaError_t LoadTextureData(std::vector<uint32_t>* rawData, std::vector<int>* offset);
 