#include "Shader.h"

void BlinnPhongShader::Vert()
{
	auto matrixM = dataTruck->matrixM;
	auto matrixVP = dataTruck->matrixVP;
	int WIDTH = dataTruck->WIDTH;
	int HEIGHT = dataTruck->HEIGHT;

	for (int i = 0; i < 3; i++)
	{
		//将positionOS转到positionWS
		dataTruck->DTpositionWS.push_back(matrixM * dataTruck->DTpositionOS[i]);
		//将positionWS转到positionCS
		dataTruck->DTpositionCS.push_back(matrixVP * dataTruck->DTpositionWS[i]);
		//将positionCS转到positionSS
		auto vertex = dataTruck->DTpositionCS[i];
		auto tmp = Eigen::Vector4f(vertex.x() * WIDTH / (2 * vertex.w()) + WIDTH / 2, vertex.y() * HEIGHT / (2 * vertex.w()) + HEIGHT / 2, vertex.z() / vertex.w(), vertex.w());
		dataTruck->DTpositionSS.push_back(tmp);
		//将normalOS转到normalWS
		auto normalos = dataTruck->DTnormalOS[i];
		Eigen::Vector4f ttmp(normalos.x(), normalos.y(), normalos.z(), 0);
		ttmp = matrixM.inverse().transpose() * ttmp;
		//将顶点uv坐标处理好
		TransformTex(&dataTruck->DTuv0, (*dataTruck->mesh->GetTextures())[0], i);
		dataTruck->DTnormalWS.push_back(ttmp.head(3));
	}
}

Eigen::Vector4f BlinnPhongShader::Frag(float a, float b, float c)
{
	// 插值出法线
	Eigen::Vector3f normal = a * dataTruck->DTnormalWS[0] + b * dataTruck->DTnormalWS[1] + c * dataTruck->DTnormalWS[2];
	normal.normalize();
	//插值出纹理坐标
	Eigen::Vector2f uv = a * dataTruck->DTuv0[0] + b * dataTruck->DTuv0[1] + c * dataTruck->DTuv0[2];
	//获取diffuse texture
	Texture* tp = (*dataTruck->mesh->GetTextures())[0];

	auto mainLight = dataTruck->mainLight;
	auto lightDir = -1 * mainLight.direction.normalized();
	float NdotL = normal.dot(lightDir);

	Eigen::Vector4f diffuse = mainLight.intensity * std::max(NdotL, 0.f) * mulColor(mainLight.color, Tex2D(tp, uv));
	
	Eigen::Vector3f worldPos = (a * dataTruck->DTpositionWS[0] + b * dataTruck->DTpositionWS[1] + c * dataTruck->DTpositionWS[2]).head(3);
	Eigen::Vector3f viewDir = (dataTruck->camera->GetPosition() - worldPos).normalized();

	Eigen::Vector3f halfDir = (viewDir + lightDir).normalized();
	Eigen::Vector4f specular = mainLight.intensity * mainLight.color * pow(std::max(0.f, normal.dot(halfDir)), 25);

	Eigen::Vector4f finalColor = diffuse + specular;
	return finalColor;
}
