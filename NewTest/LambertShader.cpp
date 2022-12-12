#include "Shader.h"
#include <iostream>
#include "GlobalSettings.h"

Texture* diabloDiffuse = new Texture("OBJs\\diablo3_pose_diffuse.tga");
Texture* diabloNormal = new Texture("OBJs\\diablo3_pose_nm_tangent.tga");
Texture* floorDiffuse = new Texture("OBJs\\floor_diffuse.tga");
Texture* floorNormal = new Texture("OBJs\\floor_nm_tangent.tga");

Varyings LambertShader::Vert(Attributes vertex)
{
	auto matrixM = vertex.matrixM;
	auto matrixVP = dataTruck->matrixVP;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	Varyings o;

	//将positionOS转到positionWS
	o.positionWS = matrixM * vertex.positionOS;
	//将positionWS转到positionCS
	o.positionCS = matrixVP * o.positionWS;

	//将normalOS转到normalWS
	Eigen::Matrix3f normalMatrix = matrixM.block(0, 0, 3, 3).inverse().transpose();
	o.normalWS = normalMatrix * vertex.normalOS;

	//将顶点uv坐标处理好
	o.uv = TransformTex(vertex.uv, diabloDiffuse);
	return o;
}

int cnt = 0;

Eigen::Vector4f LambertShader::Frag(Varyings i)
{
	//计算TBN
	/*Eigen::Vector3f v1 = (dataTruck->DTpositionCS[face.B] / dataTruck->DTpositionCS[face.B].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	Eigen::Vector3f v2 = (dataTruck->DTpositionCS[face.C] / dataTruck->DTpositionCS[face.C].w() - dataTruck->DTpositionCS[face.A] / dataTruck->DTpositionCS[face.A].w()).head(3);
	Eigen::Matrix3f A;
	A << v1.x(), v1.y(), v1.z(),
		v2.x(), v2.y(), v2.z(),
		normalWS.x(), normalWS.y(), normalWS.z();
	Eigen::Matrix3f AI = A.inverse();

	Eigen::Vector3f i = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].x() - dataTruck->DTuv0[face.A].x(), dataTruck->DTuv0[face.C].x() - dataTruck->DTuv0[face.A].x(), 0);
	Eigen::Vector3f j = AI * Eigen::Vector3f(dataTruck->DTuv0[face.B].y() - dataTruck->DTuv0[face.A].y(), dataTruck->DTuv0[face.C].y() - dataTruck->DTuv0[face.A].y(), 0);
	i.normalize();
	j.normalize();
	Eigen::Matrix3f tbnMatrix;
	tbnMatrix << i.x(), j.x(), normalWS.x(),
		i.y(), j.y(), normalWS.y(),
		i.z(), j.z(), normalWS.z();*/


	//获取diffuse texture、normal texture
	/*Texture* diffuseTex = (*dataTruck->mesh->GetTextures())[0];
	Texture* normalTex = (*dataTruck->mesh->GetTextures())[1];*/

	auto mainLight = dataTruck->mainLight;
	Eigen::Vector3f lightDirWS = -1 * mainLight.direction;
	lightDirWS.normalize();

	//获得法线纹理中法线数据
	//Eigen::Vector3f bumpTS = UnpackNormal(diabloNormal, i.uv);
	//Eigen::Vector3f bumpWS = (tbnMatrix * bumpTS).normalized();
	float NdotL = i.normalWS.dot(lightDirWS);
	//diffuse
	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * Vec4Mul(mainLight.color, Tex2D(diabloDiffuse, i.uv));
	//std::cout << "idx =  " << ++cnt << " x = " << i.uv.x() << " y = " << i.uv.y() << "color = " << Tex2D(diabloDiffuse, i.uv) << std::endl;
	float shadow = 0.f;
	if (GlobalSettings::GetInstance()->settings.drawShadow)
	{
		Eigen::Vector4f positionLSS = ComputeScreenPos(dataTruck->lightMatrixVP * i.positionWS);
		float bias = std::max(0.05 * (1 - i.normalWS.dot(lightDirWS)), 0.01);
		//PCF
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				shadow += (positionLSS.z() > dataTruck->shadowMap->GetZ(positionLSS.x() + i, positionLSS.y() + j) + bias);
			}
		}
		shadow = std::min(0.7f, shadow / 9);
	}

	Eigen::Vector4f finalColor = (1 - shadow) * diffuse;
	return finalColor;
}