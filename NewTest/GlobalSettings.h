#pragma once


struct Settings
{
	bool drawShadow;
	bool blinnPhong;
	bool lambert;
	bool drawLine;
	bool debug;
};

class GlobalSettings
{
private:
	static GlobalSettings* instance;
	
private:
	GlobalSettings();
	~GlobalSettings();
public:
	Settings settings;
	static GlobalSettings* GetInstance();
};