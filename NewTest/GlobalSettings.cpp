#include "GlobalSettings.h"

GlobalSettings::GlobalSettings()
{
	settings.blinnPhong = false;
	settings.drawLine = false;
	settings.lambert = false;
	settings.drawShadow = true;
	settings.debug = false;
}

GlobalSettings::~GlobalSettings()
{
}

GlobalSettings* GlobalSettings::GetInstance()
{
	if (instance == nullptr)
	{
		instance = new GlobalSettings();
	}
	return instance;
}

GlobalSettings* GlobalSettings::instance = nullptr;