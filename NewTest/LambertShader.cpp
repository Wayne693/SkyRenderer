#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"


Varyings LambertShader::Vert(Attributes vertex)
{
	auto matrixM = dataTruck->matrixM;
	auto matrixVP = dataTruck->matrixVP;

	Varyings o;

	//将positionOS转到positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = matrixVP * o.positionWS;
	//将normalOS转到normalWS
	Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
	o.normalWS = normalMatrix * vertex.normalOS;
	//计算tangentWS、binormalWS
	o.tangentWS = matrixM.block(0,0,3,3) * vertex.tangentOS.head(3);
	o.binormalWS = o.normalWS.cross(o.tangentWS) * vertex.tangentOS.w();
	//将顶点uv坐标处理好
	//o.uv = TransformTex(vertex.uv, &(*dataTruck->mesh->GetTextures())[0]);
	return o;
}



Eigen::Vector4f LambertShader::Frag(Varyings i)
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
	Eigen::Vector3f bumpTS = UnpackNormal(&dataTruck->textures[1], i.uv);////////////////////////todo 转cuda DONE
	Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();

	//diffuse
	float NdotL = bumpWS.dot(lightDirWS);
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * Vec4Mul(mainLight.color, Tex2D(&dataTruck->textures[0], i.uv));

	float shadow = 0.f;
	//if (GlobalSettings::GetInstance()->settings.drawShadow)////////////////////todo 要不直接删了得了
	//{
	Eigen::Vector4f positionLSS = ComputeScreenPos(dataTruck->lightMatrixVP * i.positionWS);
	float bias = std::max(0.05 * (1 - bumpWS.dot(lightDirWS)), 0.01);
	//PCF
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			//dataTruck->shadowMap.GetZ(positionLSS.x() + i, positionLSS.y() + j)
			shadow += (positionLSS.z() > GetZ(&dataTruck->shadowMap, positionLSS.x() + i, positionLSS.y() + j) + bias);////////////////////todo frameBuffer 采样改cuda DONE
		}
	}
	shadow = std::min(0.7f, shadow / 9);
	//}

	Eigen::Vector4f finalColor = (1 - shadow) * diffuse;
	//Eigen::Vector4f finalColor = Eigen::Vector4f(1, 1, 1, 1);
	return finalColor;
}