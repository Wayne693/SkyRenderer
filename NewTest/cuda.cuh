#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Shader.h"
#include "Model.h"
#include <vector>
#include <stdio.h>


cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Attributes>* fragDatas, DataTruck* dataTruck, Shader* shader);
