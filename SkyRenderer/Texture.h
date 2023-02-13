#pragma once
#include <vector>
#include <string>
#include <Dense>

//Œ∆¿Ì
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
	Texture(int width, int height);
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