#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include <math.h>
#include <iostream>
#include "StatusWindowLoop.h"
#include "Shader.cuh"
#include "RenderLoop.h"
#include "GlobalSettingWindowLoop.h"
#include "GlobalSettings.h"
#include "Pretreatment.h"
#include "Utility.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

const int WIDTH = 1280;
const int HEIGHT = 720;
Scene* mainScene;
// 窗口状态
bool show_global_window = true;
bool show_status_window = true;
//背景颜色
ImVec4 clear_color = ImVec4(0, 0, 0, 1.00f);
static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

//初始化场景
//经验光照模型
void InitSceneDiablo(Scene* mainScene);
//PBR/IBL/SkyBox
void InitSceneHelmet(Scene* mainScene);

GLFWwindow* UIInit();

void CleanUp(GLFWwindow* window)
{
	// Cleanup
	ImGui_ImplOpenGL2_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}

//将 data 加载到GPU内存
void LoadData()
{
	auto datap = TexData();
	auto offsetp = Offset();
	//纹理原始数据(RGBA)
	LoadTextureData(datap, offsetp);
	//预滤波
	LoadPrefilterMaps(dataTruck.iblMap.PrefilterMaps);
}

GLuint GLInit()
{
	GLuint renderTexture;
	glGenTextures(1, &renderTexture);//生成纹理数量，索引
	glBindTexture(GL_TEXTURE_2D, renderTexture);//将纹理设置为TEXTURE2D

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	return renderTexture;
}

void Rendering(GLFWwindow* window)
{
	// Rendering
	ImGui::Render();
	int display_w, display_h;
	glfwGetFramebufferSize(window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
	glClear(GL_COLOR_BUFFER_BIT);


	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	glfwMakeContextCurrent(window);
	glfwSwapBuffers(window);
}

int main()
{
	GLFWwindow* window = UIInit();
	GLuint renderTexture = GLInit();

	mainScene = new Scene;
	InitSceneDiablo(mainScene);
	//InitSceneHelmet(mainScene);

	LoadData();

	//最终渲染到屏幕上的FrameBuffer
	FrameBuffer* displayBuffer = new FrameBuffer(WIDTH, HEIGHT, Vector4fToColor(black));
	//shadowMap
	FrameBuffer* shadowMap = new FrameBuffer(WIDTH, HEIGHT, Vector4fToColor(black));

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL2_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//绘制状态窗口
		if (show_status_window)
		{
			StatusWindowLoop(mainScene);
		}
		//绘制设置窗口
		if (show_global_window)
		{
			GlobalSettingWindowLoop();
		}

		//渲染阴影
		if (GlobalSettings::GetInstance()->settings.drawShadow)
		{
			RenderLoop(shadowMap, shadowMap, mainScene, RENDER_SHADOW);
		}
		//渲染流程
		RenderLoop(displayBuffer, shadowMap, mainScene, RENDER_BY_PASS);

		ImTextureID imguiId = (ImTextureID)renderTexture;
		if (GlobalSettings::GetInstance()->settings.debug)
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, displayBuffer->width(), displayBuffer->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, shadowMap->GetRawBuffer());
		}
		else
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, displayBuffer->width(), displayBuffer->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, displayBuffer->GetRawBuffer());
		}
		ImDrawList* drawList = ImGui::GetBackgroundDrawList();
		drawList->AddImage(imguiId, ImVec2(0, 0), ImVec2(WIDTH, HEIGHT));


		Rendering(window);

		//数据清空
		displayBuffer->Clear(Vector4fToColor(black));
		shadowMap->Clear(Vector4fToColor(black));
	}
	CleanUp(window);
	return 0;
}

void InitSceneDiablo(Scene* mainScene)
{
	std::string fileName("OBJs\\diablo3_pose.obj");
	Mesh* diabloMesh = new Mesh(fileName);
	Model* diabloModel = new Model();
	diabloModel->SetRotation(Eigen::Vector3f(0, 160, 1));
	diabloModel->AddMesh(diabloMesh);
	diabloMesh->SetCommonShader(LAMBERT_SHADER);
	diabloMesh->SetShadowShader(SHADOWMAP_SHADER);
	Texture* diabloDiffuse = new Texture("OBJs\\diablo3_pose_diffuse.tga");
	Texture* diabloNormal = new Texture("OBJs\\diablo3_pose_nm_tangent.tga");
	diabloMesh->AddTexture(diabloDiffuse);
	diabloMesh->AddTexture(diabloNormal);
	mainScene->AddModel(diabloModel);

	fileName = "OBJs\\floor.obj";
	Mesh* floorMesh = new Mesh(fileName);
	Model* floor = new Model();
	floor->SetTranslation(Eigen::Vector3f(0, -0.12, 2));
	floor->SetTranslation(Eigen::Vector3f(0, 0, 4.360));
	floor->SetScale(Eigen::Vector3f(1.33f, 1, 1.33f));
	floor->AddMesh(floorMesh);
	floorMesh->SetCommonShader(LAMBERT_SHADER);
	floorMesh->SetShadowShader(SHADOWMAP_SHADER);
	Texture* floorDiffuse = new Texture("OBJs\\floor_diffuse.tga");
	Texture* floorNormal = new Texture("OBJs\\floor_nm_tangent.tga");
	floorMesh->AddTexture(floorDiffuse);
	floorMesh->AddTexture(floorNormal);
	mainScene->AddModel(floor);

	Camera* mainCamera = new Camera(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(0, 1, 0), 0.3f, 6.20f, 50, 1.f * WIDTH / HEIGHT);
	mainScene->AddCamera(mainCamera);

	Light mainLight;
	mainLight.direction = Eigen::Vector3f(0, -0.930f, 1);
	mainLight.color = Eigen::Vector4f(1, 1, 1, 1);
	mainLight.intensity = 1.5f;
	mainScene->SetLight(mainLight);

}

void InitSceneHelmet(Scene* mainScene)
{
	//helmet
	std::string fileName("OBJs\\helmet.obj");
	Mesh* helmetMesh = new Mesh(fileName);
	Model* helmet = new Model();
	helmet->AddMesh(helmetMesh);
	fileName = "OBJs\\helmet_albedo.tga";
	Texture* helmetAlbedo = new Texture(fileName);
	helmetMesh->AddTexture(helmetAlbedo);
	fileName = "OBJs\\helmet_normal.tga";
	Texture* helmetNormal = new Texture(fileName);
	helmetMesh->AddTexture(helmetNormal);
	fileName = "OBJs\\helmet_roughness.tga";
	Texture* helmetRoughness = new Texture(fileName);
	helmetMesh->AddTexture(helmetRoughness);
	fileName = "OBJs\\helmet_metallic.tga";
	Texture* helmetMetallic = new Texture(fileName);
	helmetMesh->AddTexture(helmetMetallic);
	fileName = "OBJs\\helmet_occlusion.tga";
	Texture* helmetOcclusion = new Texture(fileName);
	helmetMesh->AddTexture(helmetOcclusion);
	fileName = "OBJs\\helmet_emission.tga";
	Texture* helmetEmission = new Texture(fileName);
	helmetMesh->AddTexture(helmetEmission);

	helmetMesh->SetCommonShader(PBR_SHADER);
	helmetMesh->SetShadowShader(NONE);
	helmet->SetTranslation(Eigen::Vector3f(0, 0, 3.56));
	helmet->SetRotation(Eigen::Vector3f(220, 97.5, -87.5));
	mainScene->AddModel(helmet);

	//skyBox
	fileName = "OBJs\\SkyBox.obj";
	Mesh* skyBoxMesh = new Mesh(fileName);
	Model* skyBox = new Model;
	skyBox->SetTranslation(Eigen::Vector3f(0, 0, 0));
	skyBox->SetIsSkyBox(true);
	skyBox->AddMesh(skyBoxMesh);
	skyBoxMesh->SetCommonShader(SKYBOX_SHADER);
	skyBoxMesh->SetShadowShader(NONE);
	std::vector<std::string> cubemapFiles
	{
		"OBJs\\DOOMright.png",
		"OBJs\\DOOMleft.png",
		"OBJs\\DOOMtop.png",
		"OBJs\\DOOMbottom.png",
		"OBJs\\DOOMfront.png",
		"OBJs\\DOOMback.png"
	};
	CubeMap* cubeMap = new CubeMap(cubemapFiles);
	//将预处理好的irradianceMap设置给dataTruck
	dataTruck.iblMap.irradianceMap = GenerateIrradianceMap(cubeMap);
	dataTruck.iblMap.PrefilterMaps = GeneratePrefilterMap(cubeMap, 2);
	dataTruck.iblMap.level = 2;
	fileName = "OBJs\\LUT.png";
	dataTruck.iblMap.LUT = Texture(fileName);
	skyBoxMesh->SetCubeMap(cubeMap);
	mainScene->AddModel(skyBox);

	//Camera
	Camera* mainCamera = new Camera(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(0, 1, 0), 0.3f, 100, 50, 1.f * WIDTH / HEIGHT);
	mainScene->AddCamera(mainCamera);
}

GLFWwindow* UIInit()
{
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return nullptr;
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Sky Renderer", NULL, NULL);
	if (window == NULL)
		return nullptr;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL2_Init();

	return window;
}