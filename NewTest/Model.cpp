#include "Model.h"
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
#include "Stb_image/stb_image.h"

//Model
Model::Model()
{
	//½«scale¡¢translation¡¢rotation³õÊ¼»¯
	m_Scale << 1, 1, 1;
	m_Translation << 0, 0, 3;
	m_Rotation << 0, 180, 0;
	m_IsskyBox = false;
}

void Model::SetTranslation(Eigen::Vector3f translation)
{
	m_Translation = translation;
}

void Model::SetRotation(Eigen::Vector3f rotation)
{
	m_Rotation = rotation;
}

void Model::SetScale(Eigen::Vector3f scale)
{
	m_Scale = scale;
}



void Model::SetIsSkyBox(bool flag)
{
	m_IsskyBox = flag;
}

Eigen::Matrix4f Model::GetModelMatrix()
{
	return m_ModelMtx;
}

Eigen::Vector3f Model::GetTranslation()
{
	return m_Translation;
}
Eigen::Vector3f Model::GetRotation()
{
	return m_Rotation;
}
Eigen::Vector3f Model::GetScale()
{
	return m_Scale;
}

bool Model::IsSkyBox()
{
	return m_IsskyBox;
}



void Model::AddMesh(Mesh* mesh)
{
	m_Meshes.push_back(mesh);
}

std::vector<Mesh*>* Model::GetMeshes()
{
	return &m_Meshes;
}

void Model::UpdateModelMatrix()
{
	float pi = acos(-1);
	float x = m_Rotation.x() * pi / 180.0;
	float y = m_Rotation.y() * pi / 180.0;
	float z = m_Rotation.z() * pi / 180.0;
	Eigen::Matrix4f rotateX;
	rotateX << 1, 0, 0, 0,
		0, cos(x), -sin(x), 0,
		0, sin(x), cos(x), 0,
		0, 0, 0, 1;
	Eigen::Matrix4f rotateY;
	rotateY << cos(y), 0, sin(y), 0,
		0, 1, 0, 0,
		-sin(y), 0, cos(y), 0,
		0, 0, 0, 1;
	Eigen::Matrix4f rotateZ;
	rotateZ << cos(z), -sin(z), 0, 0,
		sin(z), cos(z), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;
	Eigen::Matrix4f translation;
	translation << 1, 0, 0, m_Translation.x(),
		0, 1, 0, m_Translation.y(),
		0, 0, 1, m_Translation.z(),
		0, 0, 0, 1;
	Eigen::Matrix4f scale;
	scale << m_Scale.x(), 0, 0, 0,
		0, m_Scale.y(), 0, 0,
		0, 0, m_Scale.z(), 0,
		0, 0, 0, 1;
	m_ModelMtx = translation * rotateZ * rotateX * rotateY * scale;
}


//Mesh
Mesh::Mesh(std::string fileName)
{
	std::ifstream in;
	in.open(fileName.c_str(), std::ifstream::in);
	if (in.fail())
	{
		std::cout << "Load Model Fail!" << std::endl;
	}
	std::string line;
	int num;
	while (!in.eof())
	{
		std::getline(in, line);
		if (!line.compare(0, 2, "v "))
		{
			float x, y, z;
			num = sscanf_s(line.c_str(), "v %f %f %f", &x, &y, &z);
			assert(num == 3);
			positions.push_back(Eigen::Vector4f(x, y, z, 1));
		}
		else if (!line.compare(0, 3, "vt "))
		{
			float x, y;
			num = sscanf_s(line.c_str(), "vt %f %f", &x, &y);
			assert(num == 2);
			texcoords.push_back(Eigen::Vector2f(x, y));
		}
		else if (!line.compare(0, 3, "vn "))
		{
			float x, y, z;
			num = sscanf_s(line.c_str(), "vn %f %f %f", &x, &y, &z);
			assert(num == 3);
			normals.push_back(Eigen::Vector3f(x, y, z));
		}
		else if (!line.compare(0, 2, "f "))
		{
			int x0, y0, z0, x1, y1, z1, x2, y2, z2;
			num = sscanf_s(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &x0, &y0, &z0, &x1, &y1, &z1, &x2, &y2, &z2);
			assert(num == 9);
			m_FacePositions.push_back(Face(x0, x1, x2));
			m_FaceUVs.push_back(Face(y0, y1, y2));
			m_FaceNormals.push_back(Face(z0, z1, z2));
		}
	}
}

void Mesh::SetShadowShader(Shader* shader)
{
	m_ShadowShader = shader;
}

void Mesh::SetCommonShader(Shader* shader)
{
	m_CommonShader = shader;
}

std::vector<Face>* Mesh::GetPositionFaces()
{
	return &m_FacePositions;
}

std::vector<Face>* Mesh::GetUVFaces()
{
	return &m_FaceUVs;
}

std::vector<Face>* Mesh::GetNormalFaces()
{
	return &m_FaceNormals;
}

std::vector<Eigen::Vector4f>* Mesh::GetPositions()
{
	return &positions;
}

std::vector<Eigen::Vector2f>* Mesh::GetTexcoords()
{
	return &texcoords;
}

std::vector<Eigen::Vector3f>* Mesh::GetNormals()
{
	return &normals;
}

void Mesh::SetCubeMap(CubeMap* cubeMap)
{
	m_CubeMap = cubeMap;
}

void Mesh::AddTexture(Texture* texture)
{
	m_Textures.push_back(texture);
}

std::vector<Texture*>* Mesh::GetTextures()
{
	return &m_Textures;
}

Shader* Mesh::GetShadowShader()
{
	return m_ShadowShader;
}

Shader* Mesh::GetCommonShader()
{
	return m_CommonShader;
}

CubeMap* Mesh::GetCubeMap()
{
	return m_CubeMap;
}

//Texture
Texture::Texture(std::string fileName)
{
	m_RawBuffer = (uint32_t*)stbi_load(fileName.c_str(), &m_Width, &m_Height, &m_Channel, 4);//load texture
	m_Tilling = Eigen::Vector2f(1, 1);
	m_Offset = Eigen::Vector2f(0, 0);
}

Texture::Texture(int width, int height)
{
	m_RawBuffer = (uint32_t*)malloc(sizeof(uint32_t) * width * height);
	m_Tilling = Eigen::Vector2f(1, 1);
	m_Offset = Eigen::Vector2f(0, 0);
	m_Width = width;
	m_Height = height;
	//assert(!m_RawBuffer);
}

Texture::~Texture()
{
	free(m_RawBuffer);
}

void Texture::SetTilling(Eigen::Vector2f tilling)
{
	m_Tilling = tilling;
}

void Texture::SetOffset(Eigen::Vector2f offset)
{
	m_Offset = offset;
}
//(0~1)->(0~255)
void Texture::SetData(Eigen::Vector2f uv, Eigen::Vector4f color)
{
	int x = (int)(uv.x() * m_Width);
	int y = (int)(uv.y() * m_Height);

	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Width * m_Height)
	{
		color *= 255;
		m_RawBuffer[pos] = Vector4fToColor(color);
	}
}

int Texture::width()
{
	return m_Width;
}

int Texture::height()
{
	return m_Height;
}

//(0~255)->(0~1)
Eigen::Vector4f Texture::GetData(int x, int y)
{
	int pos = (m_Height - y - 1) * m_Width + x;
	if (x >= 0 && x < m_Width && y >= 0 && y < m_Height && pos >= 0 && pos < m_Width * m_Height)
	{
		uint32_t n = m_RawBuffer[pos];
		uint8_t mask = 255;
		Eigen::Vector4f data(n & mask, (n >> 8) & mask, (n >> 16) & mask, (n >> 24) & mask);
		data /= 255.f;
		return data;
	}
	return Eigen::Vector4f(0, 0, 0, 0);
}

//repeat
Eigen::Vector4f Texture::GetData(Eigen::Vector2f uv)
{
	int x = (int)(uv.x() * m_Width);
	int y = (int)(uv.y() * m_Height);
	if (x > 0)
	{
		x = x % m_Width;
	}
	else if(x < 0)
	{
		x = m_Width + x % m_Width;
	}

	if (y > 0)
	{
		y = y % m_Height;
	}
	else if (y < 0)
	{
		y = m_Height + y % m_Height;
	}

	return GetData(x, y);
}

Eigen::Vector2f Texture::GetTilling()
{
	return m_Tilling;
}

Eigen::Vector2f Texture::GetOffset()
{
	return m_Offset;
}

//Cube Map
CubeMap::CubeMap(std::vector<std::string> fileNames)
{
	for (int i = 0; i < fileNames.size(); i++)
	{
		Texture* tmpTex = new Texture(fileNames[i]);
		m_Textures.push_back(tmpTex);
	}
}

CubeMap::CubeMap(int width,int height)
{
	for (int i = 0; i < 6; i++)
	{
		Texture* tmpTex = new Texture(width, height);
		m_Textures.push_back(tmpTex);
	}
}

CubeMap::~CubeMap()
{
	for (int i = 0; i < m_Textures.size(); i++)
	{
		if (m_Textures[i])
		{
			delete m_Textures[i];
		}
	}
}

//todo understand
int selectCubeMapFace(Eigen::Vector3f direction, Eigen::Vector2f* texcoord) {
	float abs_x = (float)fabs(direction.x());
	float abs_y = (float)fabs(direction.y());
	float abs_z = (float)fabs(direction.z());
	float ma, sc, tc;
	int face_index;

	if (abs_x > abs_y && abs_x > abs_z) {   /* major axis -> x */
		ma = abs_x;
		if (direction.x() > 0) {                  /* positive x */
			face_index = 0;
			sc = -direction.z();
			tc = -direction.y();
		}
		else {                                /* negative x */
			face_index = 1;
			sc = +direction.z();
			tc = -direction.y();
		}
	}
	else if (abs_y > abs_z) {             /* major axis -> y */
		ma = abs_y;
		if (direction.y() > 0) {                  /* positive y */
			face_index = 2;
			sc = +direction.x();
			tc = +direction.z();
		}
		else {                                /* negative y */
			face_index = 3;
			sc = +direction.x();
			tc = -direction.z();
		}
	}
	else {                                /* major axis -> z */
		ma = abs_z;
		if (direction.z() > 0) {                  /* positive z */
			face_index = 4;
			sc = +direction.x();
			tc = -direction.y();
		}
		else {                                /* negative z */
			face_index = 5;
			sc = -direction.x();
			tc = -direction.y();
		}
	}

	texcoord->x() = (sc / ma + 1) / 2;
	texcoord->y() =1 - (tc / ma + 1) / 2;
	return face_index;
}

void CubeMap::SetData(Eigen::Vector3f direction, Eigen::Vector4f col)
{
	Eigen::Vector2f uv;
	int idx = selectCubeMapFace(direction, &uv);
	m_Textures[idx]->SetData(uv, col);
}

Eigen::Vector4f CubeMap::GetData(Eigen::Vector3f direction)
{
	Eigen::Vector2f uv;
	int idx = selectCubeMapFace(direction, &uv);
	return m_Textures[idx]->GetData(uv);
}