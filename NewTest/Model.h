#pragma once
#include <vector>
#include <string>
#include "Texture.h"
#include "Dense"

//一个三角三个顶点的索引
struct Face
{
	int A;
	int B;
	int C;
	Face(int a,int b,int c)
	{
		A = a;
		B = b;
		C = c;
	}
	bool friend operator < (Face const& a ,Face const& b)
	{
		if (a.A == b.A)
		{
			if (a.B == b.B)
			{
				return a.C < b.C;
			}
			return a.B < b.B;
		}
		return a.A < b.A;
	}
};

struct Attributes
{
	Eigen::Vector4f positionOS;
	Eigen::Vector3f normalOS;
	Eigen::Vector4f tangentOS;//tangent(x,y,z) binormalsign(w)
	Eigen::Vector2f uv;
};

struct Varyings
{
	Eigen::Vector4f positionWS;
	Eigen::Vector4f positionCS;
	Eigen::Vector3f normalWS;
	Eigen::Vector3f tangentWS;//tangent(x,y,z)
	Eigen::Vector3f binormalWS;
	Eigen::Vector2f uv;
};

//网格体
class Mesh
{
public:
	std::vector<Attributes> m_VertexData;
	std::vector<Face> m_IndexData;

	std::vector<Texture> m_Textures;
	CubeMap* m_CubeMap;
	int m_ShadowShaderID;
	int m_CommonShaderID;

	Mesh(std::string filePath);
	void SetShadowShader(int shaderID);
	void SetCommonShader(int shaderID);
	void SetCubeMap(CubeMap* cubeMap);

	std::vector<Attributes>* GetVertDatas();
	std::vector<Face>* GetIndexDatas();

	void AddTexture(Texture* texture);
	std::vector<Texture>* GetTextures();
	int GetShadowShaderID();
	int GetCommonShaderID();
	CubeMap* GetCubeMap();
};


//模型
class Model
{
private:
	
	std::vector<Mesh*> m_Meshes;
	bool m_IsskyBox;				//是否为skybox

	Eigen::Vector3f m_Translation;	//平移
	Eigen::Vector3f m_Rotation;		//旋转
	Eigen::Vector3f m_Scale;		//缩放
	Eigen::Matrix4f m_ModelMtx;		//M矩阵
public:
	Model();
	void SetTranslation(Eigen::Vector3f translaton);
	void SetRotation(Eigen::Vector3f rotation);
	void SetScale(Eigen::Vector3f scale);
	
	void SetIsSkyBox(bool flag);

	Eigen::Matrix4f GetModelMatrix();
	Eigen::Vector3f GetTranslation();
	Eigen::Vector3f GetRotation();
	Eigen::Vector3f GetScale();
	bool IsSkyBox();
	
	void AddMesh(Mesh* mesh);
	std::vector<Mesh*>* GetMeshes();

	void UpdateModelMatrix();//更新M矩阵
};


