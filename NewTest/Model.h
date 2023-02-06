#pragma once
#include <vector>
#include <string>
#include "LowLevelData.h"
#include "Draw.h"


class Shader;


//һ�������������������
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


//����
class Texture
{
public:
	int m_Width;
	int m_Height;
	int m_ID;

	int m_Channel;
	Eigen::Vector2f m_Tilling;
	Eigen::Vector2f m_Offset;


	Texture(std::string fileName);
	Texture(int width,int height);
	Texture();

	void SetTilling(Eigen::Vector2f);
	void SetOffset(Eigen::Vector2f);
	void SetData(Eigen::Vector2f uv, Eigen::Vector4f color);

	int width();
	int height();
	Eigen::Vector4f GetData(int x, int y);
	Eigen::Vector4f GetData(Eigen::Vector2f uv);
	Eigen::Vector2f GetTilling();
	Eigen::Vector2f GetOffset();
};

//Cube Map
class CubeMap
{
public:
	//px nx py ny pz nz
	Texture px, nx, py, ny, pz, nz;

	CubeMap(std::vector<std::string> fileNames);
	CubeMap(int width, int height);
	CubeMap();
	void SetData(Eigen::Vector3f vector, Eigen::Vector4f col);
	Eigen::Vector4f GetData(Eigen::Vector3f vector);
};

//������
class Mesh
{
public:
	std::vector<Attributes> m_VertexData;
	std::vector<Face> m_IndexData;

	std::vector<Texture> m_Textures;
	CubeMap* m_CubeMap;
	int m_ShadowShaderID;
	int m_CommonShaderID;

	Shader* m_ShadowShader;
	Shader* m_CommonShader;

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


//ģ��
class Model
{
private:
	
	std::vector<Mesh*> m_Meshes;
	bool m_IsskyBox;				//�Ƿ�Ϊskybox

	Eigen::Vector3f m_Translation;	//ƽ��
	Eigen::Vector3f m_Rotation;		//��ת
	Eigen::Vector3f m_Scale;		//����
	Eigen::Matrix4f m_ModelMtx;		//M����
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

	void UpdateModelMatrix();//����M����
};


