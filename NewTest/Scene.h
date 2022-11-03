#pragma once
#include "Model.h"
#include "Camera.h"
#include "Dense"

struct Light
{
	Eigen::Vector3f direction;
	Eigen::Vector4f color;
	float intensity;
};

class Scene
{
private:
	std::vector<Model*> m_Models;
	std::vector<Camera*> m_Cameras;
	Light m_MainLight;

public:
	Scene();
	void AddModel(Model* model);
	void AddCamera(Camera* camera);
	void SetLight(Light light);

	std::vector<Model*>* GetModels();
	std::vector<Camera*>* GetCameras();
	Light GetLight();
};