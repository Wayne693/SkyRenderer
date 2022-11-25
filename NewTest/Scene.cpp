#include "Scene.h"
#include "iostream"
Scene::Scene()
{
	m_MainLight.direction = Eigen::Vector3f(0, 0, 1);
	m_MainLight.color = Eigen::Vector4f(1, 1, 1, 1);
	m_MainLight.intensity = 1.f;
}

void Scene::AddModel(Model* model)
{
	m_Models.push_back(model);
}

void Scene::AddCamera(Camera* camera)
{
	m_Cameras.push_back(camera);
}

void Scene::SetLight(Light light)
{
	m_MainLight = light;
}

std::vector<Model*>* Scene::GetModels()
{
	return &m_Models;
}

std::vector<Camera*>* Scene::GetCameras()
{
	return &m_Cameras;
}

Light Scene::GetLight()
{
	return m_MainLight;
}
