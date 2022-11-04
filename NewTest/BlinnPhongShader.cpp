#include "Shader.h"
#include <iostream>

void BlinnPhongShader::Vert()
{
	auto matrixM = dataTruck.matrixM;
	auto matrixVP = dataTruck.matrixVP;
	int WIDTH = dataTruck.WIDTH;
	int HEIGHT = dataTruck.HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck.DTpositionWS.push_back(matrixM * dataTruck.DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck.DTpositionCS.push_back(matrixVP * dataTruck.DTpositionWS[i]);
		//将positionCS转到positionSS
		auto vertex = dataTruck.DTpositionCS[i];
		auto tmp = Eigen::Vector4f(vertex.x() * WIDTH / (2 * vertex.w()) + WIDTH / 2, vertex.y() * HEIGHT / (2 * vertex.w()) + HEIGHT / 2, vertex.z() / vertex.w(), vertex.w());
		dataTruck.DTpositionSS.push_back(tmp);
		//将normalOS转到normalWS
		auto normalos = dataTruck.DTnormalOS[i];
		Eigen::Vector4f ttmp(normalos.x(), normalos.y(), normalos.z(), 0);
		ttmp = matrixM.inverse().transpose() * ttmp;
		dataTruck.DTnormalWS.push_back(ttmp.head(3));
	}
}

Eigen::Vector4f BlinnPhongShader::Frag(float a, float b, float c)
{
	// 插值出法线
	Eigen::Vector3f normal = a * dataTruck.DTnormalWS[0] + b * dataTruck.DTnormalWS[1] + c * dataTruck.DTnormalWS[2];
	//插值出纹理坐标
	Eigen::Vector2f uv = a * dataTruck.DTuv[0] + b * dataTruck.DTuv[1] + c * dataTruck.DTuv[2];
	//目前只有一个texture,todo:处理多个texture
	Texture* tp = (*dataTruck.model->GetTextures())[0];
	//tilling、offset后的uv坐标
	Eigen::Vector2f TFuv = Eigen::Vector2f(tp->GetTilling().x() * uv.x() * tp->width(), tp->GetTilling().y() * uv.y() * tp->height()) + tp->GetOffset();
	normal.normalize();

	auto mainLight = dataTruck.mainLight;
	auto lightDir = -1 * mainLight.direction.normalized();
	float NdotL = normal.dot(lightDir);

	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(mainLight.color, tp->GetData(TFuv));
	
	Eigen::Vector3f worldPos = (a * dataTruck.DTpositionWS[0] + b * dataTruck.DTpositionWS[1] + c * dataTruck.DTpositionWS[2]).head(3);
	Eigen::Vector3f viewDir = (dataTruck.camera->GetPosition() - worldPos).normalized();

	Eigen::Vector3f halfDir = (viewDir + lightDir).normalized();
	Eigen::Vector4f specular = mainLight.intensity * mainLight.color * pow(std::max(0.f, normal.dot(halfDir)), 25);

	Eigen::Vector4f finalColor = diffuse + specular;
	return finalColor;
}

void BlinnPhongShader::Clear()
{
	dataTruck.DTpositionOS.clear();
	dataTruck.DTpositionWS.clear();
	dataTruck.DTpositionCS.clear();
	dataTruck.DTpositionSS.clear();
	dataTruck.DTuv.clear();
	dataTruck.DTnormalOS.clear();
	dataTruck.DTnormalWS.clear();
}