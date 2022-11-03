#pragma once
#include <Dense>

class Camera
{
private:
	Eigen::Vector3f m_Position;			//�����������
	Eigen::Vector3f m_LookAt;			//���ǰ������
	Eigen::Vector3f m_Up;				//����Ϸ�����
	float m_Near;						//���ü�ƽ��
	float m_Far;						//Զ�ü�ƽ��
	float m_Fov;						//FOV
	float m_Aspect;						//��߱�

	Eigen::Matrix4f m_ViewMtx;			//V����
	Eigen::Matrix4f m_ProjectionMtx;	//P����

public:
	Camera(Eigen::Vector3f position, Eigen::Vector3f lookat, Eigen::Vector3f up, float near, float far, float fov, float aspect);
	void SetPosition(Eigen::Vector3f position);
	void SetLookAt(Eigen::Vector3f lookAt);
	void SetUp(Eigen::Vector3f up);
	void SetNearPlane(float near);
	void SetFarPlane(float far);
	void SetFov(float fov);
	void SetAspect(float aspect);
	void UpdateViewMatrix();
	void UpdateProjectionMatrix();
	void UpdateVPMatrix();

	Eigen::Vector3f GetPosition();
	Eigen::Vector3f GetLookAt();
	Eigen::Vector3f GetUp();
	Eigen::Matrix4f GetViewMatrix();
	Eigen::Matrix4f GetProjectionMatrix();
	Eigen::Matrix4f GetVPMatrix();
	float GetNearPlane();
	float GetFarPlane();
	float GetFov();
	float GetAspect();
};