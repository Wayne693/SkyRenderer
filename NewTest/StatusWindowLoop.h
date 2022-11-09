#pragma once

#include "imgui.h"
#include "Dense"
#include "Scene.h"
#include "iostream"

static inline void StatusWindowLoop(Scene* mainScene)
{
	ImGui::Begin("Status");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	if (ImGui::CollapsingHeader("Scene"))
	{
		ImGui::Text("mainLight");
		Light tmpLight = mainScene->GetLight();
		//light direction
		float direction[3] = { tmpLight.direction.x(), tmpLight.direction.y(), tmpLight.direction.z() };
		ImGui::DragFloat3("direction", direction, 0.01);
		tmpLight.direction = Eigen::Map<Eigen::Vector3f>(direction);
		//light color
		float color[4] = { tmpLight.color.x() / 255.0,tmpLight.color.y() / 255.0,tmpLight.color.z() / 255.0, tmpLight.color.w() / 255.0 };
		ImGui::ColorEdit3("color", color, ImGuiColorEditFlags_DisplayRGB);
		tmpLight.color = 255.0 * Eigen::Map<Eigen::Vector4f>(color);
		//light intensity
		float intensity = tmpLight.intensity;
		ImGui::DragFloat("intensity", &intensity, 0.01);
		tmpLight.intensity = std::max(0.f, intensity);
		mainScene->SetLight(tmpLight);

		auto modelP = mainScene->GetModels();
		auto cameraP = mainScene->GetCameras();
		ImGui::Text("%d Models %d Cameras in Scene", modelP->size(), cameraP->size());
		if (ImGui::TreeNode("Models##2"))
		{
			for (int i = 0; i < modelP->size(); i++)
			{
				auto model = (*modelP)[i];
				ImGui::Text("Model %d", i + 1);
				//平移
				Eigen::Vector3f tmp = model->GetTranslation();
				float position[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("position", position, 0.01);
				tmp = Eigen::Map<Eigen::Vector3f>(position);
				model->SetTranslation(tmp);
				//旋转
				tmp = model->GetRotation();
				float rotation[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("rotation", rotation, 0.05);
				tmp = Eigen::Map<Eigen::Vector3f>(rotation);
				model->SetRotation(tmp);
				//缩放
				tmp = model->GetScale();
				float scale[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("scale", scale, 0.01);
				tmp = Eigen::Map<Eigen::Vector3f>(scale);
				model->SetScale(tmp);
				//纹理
				auto textures = model->GetTextures();
				for (int i = 0; i < textures->size(); i++)
				{
					Texture* currentTex = (*textures)[i];
					ImGui::Text("texture %d", i + 1);
					Eigen::Vector2f tmp = currentTex->GetTilling();
					float tilling[2] = { tmp.x(),tmp.y() };
					ImGui::DragFloat2("tilling", tilling, 0.01);
					tmp = Eigen::Map<Eigen::Vector2f>(tilling);
					currentTex->SetTilling(tmp);

					tmp = currentTex->GetOffset();
					float offset[2] = { tmp.x(),tmp.y() };
					ImGui::DragFloat2("offset", offset, 0.01);
					tmp = Eigen::Map<Eigen::Vector2f>(offset);
					currentTex->SetOffset(tmp);
				}

				ImGui::Separator();
			}
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Cameras##2"))
		{
			for (int i = 0; i < cameraP->size(); i++)
			{
				auto camera = (*cameraP)[i];
				ImGui::Text("Camera %d", i + 1);

				Eigen::Vector3f tmp = camera->GetPosition();
				float position[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("position", position, 0.01);
				tmp = Eigen::Map<Eigen::Vector3f>(position);
				camera->SetPosition(tmp);

				tmp = camera->GetLookAt();
				float lookat[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("lookat", lookat, 0.01);
				tmp = Eigen::Map<Eigen::Vector3f>(lookat);
				camera->SetLookAt(tmp);

				tmp = camera->GetUp();
				float up[3] = { tmp.x(),tmp.y(),tmp.z() };
				ImGui::DragFloat3("up", up, 0.01);
				tmp = Eigen::Map<Eigen::Vector3f>(up);
				camera->SetUp(tmp);

				float tmpf = camera->GetNearPlane();
				ImGui::DragFloat("NearPlane", &tmpf, 0.01);
				camera->SetNearPlane(tmpf);

				tmpf = camera->GetFarPlane();
				ImGui::DragFloat("FarPlane", &tmpf, 0.01);
				camera->SetFarPlane(tmpf);

				tmpf = camera->GetFov();
				ImGui::DragFloat("FOV", &tmpf, 0.05);
				camera->SetFov(tmpf);

				tmpf = camera->GetAspect();
				ImGui::DragFloat("Aspect", &tmpf, 0.01);
				camera->SetAspect(tmpf);
				ImGui::Separator();
			}
			ImGui::TreePop();
		}
	}
	ImGui::End();
}
