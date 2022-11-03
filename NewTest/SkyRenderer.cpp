#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include "Draw.h"
#include "Model.h"
#include "Camera.h"
#include "Scene.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "StatusWindowLoop.h"
#include "RenderLoop.h"

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
	bool show_demo_window = true;
	bool show_another_window = false;
	bool show_status_window = true;
	bool show_Draw_Window = true;
	//背景颜色
	ImVec4 clear_color = ImVec4(0, 0, 0, 1.00f);


	GLuint renderTexture;
	glGenTextures(1, &renderTexture);//生成纹理数量，索引
	glBindTexture(GL_TEXTURE_2D, renderTexture);//将纹理设置为TEXTURE2D

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	mainScene = new Scene();
	//std::string fileName("OBJs\\african_head.obj");
	std::string fileName("OBJs\\diablo3_pose.obj");
	Model african_head(fileName);
	african_head.SetTranslation(Eigen::Vector3f(0, 0, -3.f));
	african_head.SetRotation(Eigen::Vector3f(0, 0, 0));
	//fileName = "OBJs\\african_head_diffuse.tga";
	fileName = "OBJs\\diablo3_pose_diffuse.tga";
	Texture african_diffuse(fileName);
	african_head.AddTexture(&african_diffuse);
	mainScene->AddModel(&african_head);
	Camera mainCamera(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, -1), Eigen::Vector3f(0, 1, 0), 0.3f, 100, 60, 1.f * WIDTH / HEIGHT);
	mainScene->AddCamera(&mainCamera);

	FrameBuffer* frameBuffer = new FrameBuffer(WIDTH, HEIGHT, Vector4fToColor(black));

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

		//渲染流程
		RenderLoop(renderTexture,frameBuffer,mainScene);

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

	// Cleanup
	ImGui_ImplOpenGL2_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
