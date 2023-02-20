#pragma once
#include <Dense>
#include <vector>

class Camera
{
public:
	Eigen::Vector3f m_Position;							//相机世界坐标
	Eigen::Vector3f m_LookAt;							//相机前方方向(world space)
	Eigen::Vector3f m_AsixY;							//相机Y轴
	float m_Near;										//近裁剪平面
	float m_Far;										//远裁剪平面
	float m_Fov;										//FOV
	float m_Aspect;										//宽高比
	float m_Size;										//正交相机Size

	Eigen::Matrix4f m_ViewMtx;							//V矩阵
	Eigen::Matrix4f m_ProjectionMtx;					//P矩阵
	Eigen::Matrix4f m_OrthoMtx;							//正交P矩阵
	std::vector<Eigen::Vector3f> m_VisualCone;			//相机视锥体(world space)

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