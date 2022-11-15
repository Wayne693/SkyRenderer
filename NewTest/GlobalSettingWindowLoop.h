#pragma once
#include "GlobalSettings.h"
#include "imgui.h"
static inline void GlobalSettingWindowLoop()
{
	GlobalSettings* gs = GlobalSettings::GetInstance();

	ImGui::Begin("GlobalSettings");
	ImGui::Checkbox("Shadow", &(gs->GetInstance()->settings.drawShadow));
	ImGui::SameLine();
	ImGui::Checkbox("Specular", &(gs->GetInstance()->settings.blinnPhong));
	ImGui::SameLine();
	ImGui::Checkbox("Debug", &(gs->GetInstance()->settings.debug));
	ImGui::End();
}