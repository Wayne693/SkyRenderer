#include "Camera.h"

Camera::Camera(Eigen::Vector3f position, Eigen::Vector3f lookat, Eigen::Vector3f up, float near, float far, float fov, float aspect) :
	m_Position(position),
	m_LookAt(lookat),
	m_Up(up),
	m_Near(near),
	m_Far(far),
	m_Fov(fov),
	m_Aspect(aspect)
{

}

Camera::Camera()
{
}

void Camera::SetPosition(Eigen::Vector3f position)
{
	m_Position = position;
}

void Camera::SetLookAt(Eigen::Vector3f lookAt)
{
	m_LookAt = lookAt;
}

void Camera::SetUp(Eigen::Vector3f up)
{
	m_Up = up;
}

void Camera::SetNearPlane(float near)
{
	m_Near = near;
}

void Camera::SetFarPlane(float far)
{
	m_Far = far;
}

void Camera::SetFov(float fov)
{
	m_Fov = fov;
}

void Camera::SetAspect(float aspect)
{
	m_Aspect = aspect;
}

void Camera::SetSize(float size)
{
	m_Size = size;
}

void Camera::UpdateViewMatrix()
{
	Eigen::Matrix4f linerTrans;
	Eigen::Vector3f lookAt = m_LookAt.normalized();
	Eigen::Vector3f up = m_Up.normalized();
	Eigen::Vector3f asixX = lookAt.cross(up);
	linerTrans << asixX.x(), asixX.y(), asixX.z(), 0,
		up.x(), up.y(), up.z(), 0,
		-lookAt.x(), -lookAt.y(), -lookAt.z(), 0,
		0, 0, 0, 1;
	
	Eigen::Matrix4f translation;
	translation << 1, 0, 0, -m_Position.x(),
		0, 1, 0, -m_Position.y(),
		0, 0, 1, -m_Position.z(),
		0, 0, 0, 1;
	m_ViewMtx =  linerTrans * translation;
}

void Camera::UpdateProjectionMatrix()
{
	float pi = acos(-1);
	float halfFov = m_Fov * pi / 360.0;
	m_ProjectionMtx << 1.0 / (tan(halfFov) * m_Aspect), 0, 0, 0,
		0, 1.0 / tan(halfFov), 0, 0,
		0, 0, -(m_Far + m_Near) / (m_Far - m_Near), -(2 * m_Near * m_Far) / (m_Far - m_Near),
		0, 0, -1, 0;
}

void Camera::UpdateOrthoMatrix()
{
	m_OrthoMtx << 1.f / (m_Aspect * m_Size), 0, 0, 0,
		0, 1 / m_Size, 0, 0,
		0, 0, -2 / (m_Far - m_Near), -(m_Far + m_Near) / (m_Far - m_Near),
		0, 0, 0, 1;
}

void Camera::UpdateVPMatrix()
{
	Camera::UpdateViewMatrix();
	Camera::UpdateProjectionMatrix();
}

void Camera::UpdateOrthoVPMatrix()
{
	Camera::UpdateViewMatrix();
	Camera::UpdateOrthoMatrix();
}

Eigen::Matrix4f Camera::GetViewMatrix()
{
	return m_ViewMtx;
}

Eigen::Matrix4f Camera::GetProjectionMatrix()
{
	return m_ProjectionMtx;
}

Eigen::Matrix4f Camera::GetOrthoMatrix()
{
	return m_OrthoMtx;
}

Eigen::Matrix4f Camera::GetVPMatrix()
{
	return m_ProjectionMtx * m_ViewMtx;
}

Eigen::Matrix4f Camera::GetOrthoVPMatrix()
{
	return m_OrthoMtx * m_ViewMtx;
}

//Ã¿Ö¡¸üÐÂ
void Camera::CalculateVisualCone()
{
	m_VisualCone.clear();
	float pi = acos(-1);
	float halfFov = m_Fov * pi / 360.0;

	Eigen::Vector3f toNear = m_Near * m_LookAt.normalized();
	Eigen::Vector3f toTop = m_Near * tan(halfFov) * m_Up.normalized();
	Eigen::Vector3f toRight = m_Aspect * m_Near * tan(halfFov) * m_LookAt.cross(m_Up).normalized();
	Eigen::Vector3f nearPos = m_Position + toNear;

	m_VisualCone.push_back(nearPos + toTop + toRight);//NearTopRight
	m_VisualCone.push_back(nearPos + toTop - toRight);//NearTopleft
	m_VisualCone.push_back(nearPos - toTop + toRight);//NearBottomRight
	m_VisualCone.push_back(nearPos - toTop - toRight);//NearBottomleft

	Eigen::Vector3f toFar = m_Far * m_LookAt.normalized();
	toTop = m_Far * tan(halfFov) * m_Up.normalized();
	toRight = m_Aspect * m_Far * tan(halfFov) * m_LookAt.cross(m_Up).normalized();
	Eigen::Vector3f farPos = m_Position + toFar;

	m_VisualCone.push_back(farPos + toTop + toRight);//farTopRight
	m_VisualCone.push_back(farPos + toTop - toRight);//farTopleft
	m_VisualCone.push_back(farPos - toTop + toRight);//farBottomRight
	m_VisualCone.push_back(farPos - toTop - toRight);//farBottomleft
}


std::vector<Eigen::Vector3f>* Camera::GetVisualCone()
{
	return &m_VisualCone;
}

Eigen::Vector3f Camera::GetPosition()
{
	return m_Position;
}

Eigen::Vector3f Camera::GetLookAt()
{
	return m_LookAt;
}

Eigen::Vector3f Camera::GetUp()
{
	return m_Up;
}

float Camera::GetNearPlane()
{
	return m_Near;
}

float Camera::GetFarPlane()
{
	return m_Far;
}

float Camera::GetFov()
{
	return m_Fov;
}

float Camera::GetAspect()
{
	return m_Aspect;
}

float Camera::GetSize()
{
	return m_Size;
}
