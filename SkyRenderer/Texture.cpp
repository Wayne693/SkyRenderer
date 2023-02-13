#include "Texture.h"
#include "DataPool.h"
#include "Sampling.h"

#define STB_IMAGE_IMPLEMENTATION
#include "Stb_image/stb_image.h"

//Texture
Texture::Texture(std::string fileName)
{
	auto rawdata = (uint32_t*)stbi_load(fileName.c_str(), &m_Width, &m_Height, &m_Channel, 4);//load texture
	m_ID = AddTextureData(rawdata, m_Width * m_Height);
	m_Tilling = Eigen::Vector2f(1, 1);
	m_Offset = Eigen::Vector2f(0, 0);
}

Texture::Texture(int width, int height)
{
	auto rawdata = (uint32_t*)malloc(sizeof(uint32_t) * width * height);
	m_ID = AddTextureData(rawdata, width * height);
	m_Tilling = Eigen::Vector2f(1, 1);
	m_Offset = Eigen::Vector2f(0, 0);
	m_Width = width;
	m_Height = height;
}

Texture::Texture()
{

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
		SetTexData(m_ID, pos, Vector4fToColor(color));
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
		uint32_t n = GetTexData(m_ID, pos);
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
	else if (x < 0)
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
	if (fileNames.size() != 6)
	{
		printf("Create CubeMap Failed : SizeError");
		return;
	}
	px = Texture(fileNames[0]);
	nx = Texture(fileNames[1]);
	py = Texture(fileNames[2]);
	ny = Texture(fileNames[3]);
	pz = Texture(fileNames[4]);
	nz = Texture(fileNames[5]);
}

CubeMap::CubeMap(int width, int height)
{
	px = Texture(width, height);
	nx = Texture(width, height);
	py = Texture(width, height);
	ny = Texture(width, height);
	pz = Texture(width, height);
	nz = Texture(width, height);
}

CubeMap::CubeMap()
{
}




void CubeMap::SetData(Eigen::Vector3f direction, Eigen::Vector4f col)
{
	Eigen::Vector2f uv;
	int idx = selectCubeMapFace(direction, &uv);
	Texture* tmp = &px;
	switch (idx)
	{
	case 0:
		break;
	case 1:
		tmp = &nx;
		break;
	case 2:
		tmp = &py;
		break;
	case 3:
		tmp = &ny;
		break;
	case 4:
		tmp = &pz;
		break;
	case 5:
		tmp = &nz;
		break;
	}
	tmp->SetData(uv, col);
}

Eigen::Vector4f CubeMap::GetData(Eigen::Vector3f direction)
{
	Eigen::Vector2f uv;
	int idx = selectCubeMapFace(direction, &uv);
	Texture* tmp = &px;
	switch (idx)
	{
	case 0:
		break;
	case 1:
		tmp = &nx;
		break;
	case 2:
		tmp = &py;
		break;
	case 3:
		tmp = &ny;
		break;
	case 4:
		tmp = &pz;
		break;
	case 5:
		tmp = &nz;
		break;
	}
	return tmp->GetData(uv);
}