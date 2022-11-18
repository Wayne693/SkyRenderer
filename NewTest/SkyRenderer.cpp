#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Draw.h"
#include "Model.h"
#include "Camera.h"
#include "Scene.h"
#include "Shader.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "StatusWindowLoop.h"
#include "RenderLoop.h"
#include "GlobalSettingWindowLoop.h"
#include "GlobalSettings.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

const int WIDTH = 1280;
const int HEIGHT = 720;
Scene* mainScene;

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

//初始化场景
void InitScene(Scene* mainScene)
{
	std::string fileName("OBJs\\diablo3_pose.obj");
	Mesh* diabloMesh = new Mesh(fileName);
	Model* diabloModel = new Model();
	diabloModel->SetRotation(Eigen::Vector3f(0, 160, 1));
	diabloModel->AddMesh(diabloMesh);
	fileName = "OBJs\\diablo3_pose_diffuse.tga";
	Texture* diabloDiffuse = new Texture(fileName);
	fileName = "OBJs\\diablo3_pose_nm_tangent.tga";
	Texture* diabloNormal = new Texture(fileName);
	diabloModel->AddTexture(diabloDiffuse);
	diabloModel->AddTexture(diabloNormal);
	mainScene->AddModel(diabloModel);

	fileName = "OBJs\\floor.obj";
	Mesh* floorMesh = new Mesh(fileName);
	Model* floor = new Model();
	floor->SetTranslation(Eigen::Vector3f(0, 0, 4.360));
	floor->SetScale(Eigen::Vector3f(1.33f, 1, 1.33f));
	floor->AddMesh(floorMesh);
	fileName = "OBJs\\floor_diffuse.tga";
	Texture* floorDiffuse = new Texture(fileName);
	floor->AddTexture(floorDiffuse);
	fileName = "OBJs\\floor_nm_tangent.tga";
	Texture* floorNormal = new Texture(fileName);
	floor->AddTexture(floorNormal);
	mainScene->AddModel(floor);

	Camera* mainCamera = new Camera(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(0, 1, 0), 0.3f, 6.20f, 50, 1.f * WIDTH / HEIGHT);
	mainScene->AddCamera(mainCamera);

	Light mainLight;
	mainLight.direction = Eigen::Vector3f(0, -0.930f, 1);
	mainLight.color = Eigen::Vector4f(255, 255, 255, 255);
	mainLight.intensity = 1.5f;
	mainScene->SetLight(mainLight);
}

int main()
{
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Sky Renderer", NULL, NULL);
	if (window == NULL)
		return 1;
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

	// 窗口状态
	bool show_global_window = true;
	bool show_status_window = true;
	//背景颜色
	ImVec4 clear_color = ImVec4(0, 0, 0, 1.00f);


	GLuint renderTexture;
	glGenTextures(1, &renderTexture);//生成纹理数量，索引
	glBindTexture(GL_TEXTURE_2D, renderTexture);//将纹理设置为TEXTURE2D

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	mainScene = new Scene;
	InitScene(mainScene);
	
	//最终渲染到屏幕上的FrameBuffer
	FrameBuffer* displayBuffer = new FrameBuffer(WIDTH, HEIGHT, Vector4fToColor(black));
	//shadowMap
	FrameBuffer* shadowMap = new FrameBuffer(WIDTH ,HEIGHT , Vector4fToColor(black));

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

		if (show_global_window)
		{
			GlobalSettingWindowLoop();
		}

		BlinnPhongShader bpshader;
		LambertShader lshader;
		NormalMapShader nmshader;
		ShadowMapShader smshader;

		Eigen::Vector3f lightMatrixVP;
		//如果要渲染阴影
		if (GlobalSettings::GetInstance()->settings.drawShadow)
		{
			RenderLoop(shadowMap, shadowMap, mainScene, &smshader);
			nmshader.dataTruck.lightMatrixVP = smshader.dataTruck.lightMatrixVP;
			lshader.dataTruck.lightMatrixVP = smshader.dataTruck.lightMatrixVP;
		}
		
		//渲染流程
		if (GlobalSettings::GetInstance()->settings.blinnPhong)
		{
			RenderLoop(displayBuffer, shadowMap, mainScene, &nmshader);
		}
		else
		{
			RenderLoop(displayBuffer, shadowMap, mainScene, &lshader);
		}
		

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
		
		//数据清空
		displayBuffer->Clear(Vector4fToColor(black));
		shadowMap->Clear(Vector4fToColor(black));
	}

	// Cleanup
	ImGui_ImplOpenGL2_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
