#include <vector>
#include <stdio.h>
#include "cuda_runtime.h"
#include "DataTruck.h"


//��������&���ú˺���&�ͷ��ڴ�
cudaError_t VertKernel(std::vector<Attributes>* vertDatas, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID);

cudaError_t FragKernel(FrameBuffer frameBuffer, std::vector<Varyings>* fragDatas, DataTruck* dataTruck, int shaderID);




 