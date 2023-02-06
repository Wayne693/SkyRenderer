#include <vector>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Shader.h"
//#include "DataPool.h"


__host__ __device__ Eigen::Vector3f barycentric(Eigen::Vector2f A, Eigen::Vector2f B, Eigen::Vector2f C, Eigen::Vector2f P);

//加载数据与调用核函数与释放内存
cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID);

cudaError_t FragKernel(FrameBuffer frameBuffer, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID);

//加载预滤波环境贴图
cudaError_t LoadPrefilterMaps(std::vector<CubeMap>* prefilterMaps);


 