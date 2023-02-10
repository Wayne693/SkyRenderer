#include "GlobalSettings.h"

GlobalSettings::GlobalSettings()
{
	//settings.blinnPhong = false;
	//settings.drawLine = false;
	//settings.lambert = false;
	settings.drawShadow = false;
	settings.debug = false;
	//settings.skyBox = true;
}

GlobalSettings::~GlobalSettings()
{
}

GlobalSettings* GlobalSettings::GetInstance()
{
	if (!instance)
	{
		instance = new GlobalSettings();
	}
	return instance;
}

GlobalSettings* GlobalSettings::instance = nullptr;