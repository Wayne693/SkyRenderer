#pragma once
#include <vector>
#include <string>
#include "Dense"
#include "Draw.h"

class Shader;

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
};

//enum TextureFormat
//{
//	FHDR,
//	FLDR
//};

//纹理
class Texture
{
private:
	int m_Width;
	int m_Height;
	int m_Channel;
	Eigen::Vector2f m_Tilling;
	Eigen::Vector2f m_Offset;
	uint32_t* m_RawBuffer;
public:
	Texture(std::string fileName);
	Texture(int width,int height);
	~Texture();
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
	std::vector<Texture*> m_Textures;

	CubeMap(std::vector<std::string> fileNames);
	CubeMap(int width, int height);
	~CubeMap();
	void SetData(Eigen::Vector3f vector, Eigen::Vector4f col);
	Eigen::Vector4f GetData(Eigen::Vector3f vector);
};

//网格体
class Mesh
{
private:
	std::vector<Eigen::Vector4f> positions;
	std::vector<Eigen::Vector2f> texcoords;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3f> tangents;
	std::vector<Face> m_FacePositions;
	std::vector<Face> m_FaceUVs;
	std::vector<Face> m_FaceNormals;

	std::vector<Texture*> m_Textures;
	CubeMap* m_CubeMap;
	Shader* m_ShadowShader;
	Shader* m_CommonShader;
public:
	Mesh(std::string filePath);
	void SetShadowShader(Shader* shader);
	void SetCommonShader(Shader* shader);
	void SetCubeMap(CubeMap* cubeMap);

	void AddTexture(Texture* texture);

	std::vector<Face>* GetPositionFaces();
	std::vector<Face>* GetUVFaces();
	std::vector<Face>* GetNormalFaces();
	
	std::vector<Eigen::Vector4f>* GetPositions();
	std::vector<Eigen::Vector2f>* GetTexcoords();
	std::vector<Eigen::Vector3f>* GetNormals();
	std::vector<Texture*>* GetTextures();
	Shader*  GetShadowShader();
	Shader*  GetCommonShader();
	CubeMap* GetCubeMap();
};


//模型
class Model
{
private:
	
	std::vector<Mesh*> m_Meshes;
	bool m_IsskyBox;//todo 看看有没有用

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


