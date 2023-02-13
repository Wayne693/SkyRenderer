#include "Model.h"
#include "mikktspace.h"
#include "DataPool.h"
#include <fstream>
#include <map>
#include <iostream>



//Model
Model::Model()
{
	//将scale、translation、rotation初始化
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

std::vector<Eigen::Vector4f> positions;
std::vector<Eigen::Vector2f> texcoords;
std::vector<Eigen::Vector3f> normals;
std::map<Face, int> mp;
int faceNum = 0;

//实现计算切线的接口
struct VIData
{
	std::vector<Attributes>* vertDatas;
	std::vector<Face>* indexDatas;
};
int GetNumFaces(const SMikkTSpaceContext* pContext)
{
	return faceNum;
}
int GetNumVerts(const SMikkTSpaceContext* pContext, const int iFace)
{
	return 3;
}
void GetPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert)
{
	VIData* vidata = (VIData*)pContext->m_pUserData;
	Face face = (*vidata->indexDatas)[iFace];
	Eigen::Vector3f pos;
	if (iVert == 0)
	{
		pos = (*vidata->vertDatas)[face.A].positionOS.head(3);
	}
	else if (iVert == 1)
	{
		pos = (*vidata->vertDatas)[face.B].positionOS.head(3);
	}
	else if (iVert == 2)
	{
		pos = (*vidata->vertDatas)[face.C].positionOS.head(3);
	}
	fvPosOut[0] = pos.x();
	fvPosOut[1] = pos.y();
	fvPosOut[2] = pos.z();
}
void GetNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert)
{
	VIData* vidata = (VIData*)pContext->m_pUserData;
	Face face = (*vidata->indexDatas)[iFace];
	Eigen::Vector3f norm;
	if (iVert == 0)
	{
		norm = (*vidata->vertDatas)[face.A].normalOS;
	}
	else if (iVert == 1)
	{
		norm = (*vidata->vertDatas)[face.B].normalOS;
	}
	else if (iVert == 2)
	{
		norm = (*vidata->vertDatas)[face.C].normalOS;
	}
	fvNormOut[0] = norm.x();
	fvNormOut[1] = norm.y();
	fvNormOut[2] = norm.z();
}
void GetTexcoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert)
{
	VIData* vidata = (VIData*)pContext->m_pUserData;
	Face face = (*vidata->indexDatas)[iFace];
	Eigen::Vector2f uv;
	if (iVert == 0)
	{
		uv = (*vidata->vertDatas)[face.A].uv;
	}
	else if (iVert == 1)
	{
		uv = (*vidata->vertDatas)[face.B].uv;
	}
	else if (iVert == 2)
	{
		uv = (*vidata->vertDatas)[face.C].uv;
	}
	fvTexcOut[0] = uv.x();
	fvTexcOut[1] = uv.y();
}
void SetTSpace(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert)
{
	VIData* vidata = (VIData*)pContext->m_pUserData;
	Face face = (*vidata->indexDatas)[iFace];
	if (iVert == 0)
	{
		(*vidata->vertDatas)[face.A].tangentOS.x() = fvTangent[0];
		(*vidata->vertDatas)[face.A].tangentOS.y() = fvTangent[1];
		(*vidata->vertDatas)[face.A].tangentOS.z() = fvTangent[2];
		(*vidata->vertDatas)[face.A].tangentOS.w() = fSign;
	}
	else if (iVert == 1)
	{
		(*vidata->vertDatas)[face.B].tangentOS.x() = fvTangent[0];
		(*vidata->vertDatas)[face.B].tangentOS.y() = fvTangent[1];
		(*vidata->vertDatas)[face.B].tangentOS.z() = fvTangent[2];
		(*vidata->vertDatas)[face.B].tangentOS.w() = fSign;
	}
	else if (iVert == 2)
	{
		(*vidata->vertDatas)[face.C].tangentOS.x() = fvTangent[0];
		(*vidata->vertDatas)[face.C].tangentOS.y() = fvTangent[1];
		(*vidata->vertDatas)[face.C].tangentOS.z() = fvTangent[2];
		(*vidata->vertDatas)[face.C].tangentOS.w() = fSign;
	}
}

void InitMeshContainer()
{
	positions.clear();
	texcoords.clear();
	normals.clear();
	mp.clear();
	faceNum = 0;
}

//计算切线数据
void CaculateTangent(Mesh* mesh)
{
	SMikkTSpaceContext tancon;
	VIData vidata;
	SMikkTSpaceInterface ssi;
	vidata.vertDatas = &mesh->m_VertexData;
	vidata.indexDatas = &mesh->m_IndexData;
	tancon.m_pUserData = &vidata;
	ssi.m_getNumFaces = GetNumFaces;
	ssi.m_getNumVerticesOfFace = GetNumVerts;
	ssi.m_getPosition = GetPosition;
	ssi.m_getNormal = GetNormal;
	ssi.m_getTexCoord = GetTexcoord;
	ssi.m_setTSpaceBasic = SetTSpace;
	ssi.m_setTSpace = nullptr;
	tancon.m_pInterface = &ssi;
	genTangSpaceDefault(&tancon);
}

void VertAssembling(Attributes* vert, int a, int b, int c)
{
	vert->positionOS = positions[a - 1];
	vert->uv = texcoords[b - 1];
	vert->normalOS = normals[c - 1];
}



/*
* Mesh
* 加载时将数据处理为VertData和IndexData
*/
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
	int cnt = 0;
	InitMeshContainer();
	//将obj文件中数据处理好
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
			Attributes vertA, vertB, vertC;

			VertAssembling(&vertA, x0, y0, z0);
			VertAssembling(&vertB, x1, y1, z1);
			VertAssembling(&vertC, x2, y2, z2);

			Face tmpA(x0 - 1, y0 - 1, z0 - 1);
			Face trangle(0, 0, 0);

			if (mp.find(tmpA) == mp.end())
			{
				mp[tmpA] = cnt++;
				m_VertexData.push_back(vertA);
			}
			trangle.A = mp[tmpA];

			Face tmpB(x1 - 1, y1 - 1, z1 - 1);
			if (mp.find(tmpB) == mp.end())
			{
				mp[tmpB] = cnt++;
				m_VertexData.push_back(vertB);
			}
			trangle.B = mp[tmpB];

			Face tmpC(x2 - 1, y2 - 1, z2 - 1);
			if (mp.find(tmpC) == mp.end())
			{
				mp[tmpC] = cnt++;
				m_VertexData.push_back(vertC);
			}
			trangle.C = mp[tmpC];

			m_IndexData.push_back(trangle);
			faceNum++;
		}
	}
	CaculateTangent(this);
}

void Mesh::SetShadowShader(int shaderID)
{
	m_ShadowShaderID = shaderID;
}

void Mesh::SetCommonShader(int shaderID)
{
	m_CommonShaderID = shaderID;
}

void Mesh::SetCubeMap(CubeMap* cubeMap)
{
	m_CubeMap = cubeMap;
}


std::vector<Attributes>* Mesh::GetVertDatas()
{
	return &m_VertexData;
}

std::vector<Face>* Mesh::GetIndexDatas()
{
	return &m_IndexData;
}

void Mesh::AddTexture(Texture* texture)
{
	m_Textures.push_back(*texture);
}

std::vector<Texture>* Mesh::GetTextures()
{
	return &m_Textures;
}

int Mesh::GetShadowShaderID()
{
	return m_ShadowShaderID;
}

int Mesh::GetCommonShaderID()
{
	return m_CommonShaderID;
}

CubeMap* Mesh::GetCubeMap()
{
	return m_CubeMap;
}

