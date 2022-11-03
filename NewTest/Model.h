#pragma once
#include <vector>
#include <string>
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
};

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
	~Texture();
	void SetTilling(Eigen::Vector2f);
	void SetOffset(Eigen::Vector2f);

	int width();
	int height();
	Eigen::Vector4f GetData(int x, int y);
	Eigen::Vector4f GetData(Eigen::Vector2f uv);
	Eigen::Vector2f GetTilling();
	Eigen::Vector2f GetOffset();
};

class Model
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

	Eigen::Vector3f m_Translation;	//平移
	Eigen::Vector3f m_Rotation;		//旋转
	Eigen::Vector3f m_Scale;		//缩放
	Eigen::Matrix4f m_ModelMtx;		//M矩阵
public:
	Model(std::string filePath);
	void SetTranslation(Eigen::Vector3f translaton);
	void SetRotation(Eigen::Vector3f rotation);
	void SetScale(Eigen::Vector3f scale);

	Eigen::Matrix4f GetModelMatrix();
	std::vector<Face>* GetPositionFaces();
	std::vector<Face>* GetUVFaces();
	std::vector<Face>* GetNormalFaces();

	std::vector<Eigen::Vector4f>* GetPositions();
	std::vector<Eigen::Vector2f>* GetTexcoords();
	std::vector<Eigen::Vector3f>* GetNormals();

	Eigen::Vector3f GetTranslation();
	Eigen::Vector3f GetRotation();
	Eigen::Vector3f GetScale();

	void AddTexture(Texture* texture);
	std::vector<Texture*>* GetTextures();

	void UpdateModelMatrix();//更新M矩阵
};


