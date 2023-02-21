#pragma once
#include <Dense>
#include <vector>

class Camera
{
public:
	Eigen::Vector3f m_Position;							//�����������
	Eigen::Vector3f m_LookAt;							//���ǰ������(world space)
	Eigen::Vector3f m_AsixY;							//���Y��
	float m_Near;										//���ü�ƽ��
	float m_Far;										//Զ�ü�ƽ��
	float m_Fov;										//FOV
	float m_Aspect;										//��߱�
	float m_Size;										//�������Size

	Eigen::Matrix4f m_ViewMtx;							//V����
	Eigen::Matrix4f m_ProjectionMtx;					//P����
	Eigen::Matrix4f m_OrthoMtx;							//����P����
	std::vector<Eigen::Vector3f> m_VisualCone;			//�����׶��(world space)

public:
	Camera(Eigen::Vector3f position, Eigen::Vector3f lookat, float near, float far, float fov, float aspect);
	Camera();
	void SetPosition(Eigen::Vector3f position);
	void SetLookAt(Eigen::Vector3f lookAt);
	void SetNearPlane(float near);
	void SetFarPlane(float far);
	void SetFov(float fov);
	void SetAspect(float aspect);
	void SetSize(float size);
	void UpdateViewMatrix();
	void UpdateProjectionMatrix();
	void UpdateOrthoMatrix();
	void UpdateVPMatrix();
	void UpdateOrthoVPMatrix();

	Eigen::Vector3f GetPosition();
	Eigen::Vector3f GetLookAt();
	Eigen::Matrix4f GetViewMatrix();
	Eigen::Matrix4f GetProjectionMatrix();
	Eigen::Matrix4f GetOrthoMatrix();
	Eigen::Matrix4f GetVPMatrix();
	Eigen::Matrix4f GetOrthoVPMatrix();
	void CalculateVisualCone();
	std::vector<Eigen::Vector3f>* GetVisualCone();

	float GetNearPlane();
	float GetFarPlane();
	float GetFov();
	float GetAspect();
	float GetSize();
};