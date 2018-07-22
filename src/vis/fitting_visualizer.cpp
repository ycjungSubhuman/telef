#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstddef>
#include <Eigen/Core>
#define _USE_MATH_DEFINES
#include <cmath>

#include "vis/fitting_visualizer.h"
#include "mesh/colormapping.h"

namespace {
    using namespace telef::vis;
    using namespace telef::mesh;

    inline void killGlfw(std::string msg) {
        glfwTerminate();
        throw std::runtime_error(msg);
    }

    void init() {
        if(!glfwInit()) {
            killGlfw("Error on GLFW initialization");
        }
    }

    void mouseButtonCallbackGLFW(GLFWwindow *window, int button, int action, int mods) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->mouseButtonCallbackGLFW(window, button, action, mods);
    }

    void mousePositionCallbackGLFW(GLFWwindow *window, double xpos, double ypos) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->mousePositionCallback(window, xpos, ypos);
    }

    void mouseScrollCallbackGLFW(GLFWwindow *window, double xoffset, double yoffset) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->mouseScrollCallback(window, xoffset, yoffset);
    }

    GLFWwindow *getWindow (FittingVisualizer *visualizer) {
        auto window = glfwCreateWindow(1920, 1080, "Fitting Visualizer", NULL, NULL);
        if(!window) {
            killGlfw("Error on GLFW window creation");
        }
        glfwSetWindowUserPointer(window, visualizer);

        // Register input callbacks
        glfwSetMouseButtonCallback(window, mouseButtonCallbackGLFW);
        glfwSetCursorPosCallback(window, mousePositionCallbackGLFW);
        glfwSetScrollCallback(window, mouseScrollCallbackGLFW);

        glfwMakeContextCurrent(window);
        if(glewInit() != GLEW_OK) {
            throw std::runtime_error("GLEW init fail");
        }
        glfwSwapInterval(1);
        return window;
    }

    static const char *pointcloud_vertex_shader =
            "#version 460 \n"
            "uniform mat4 mvp; \n"
            "in vec4 pos; \n"
            "in float _rgb; \n"
            "out vec3 color; \n"
            "void main() { \n"
            "  gl_Position = mvp * pos; \n"
            "  gl_PointSize = 2.0; \n"
            "  float r = float((floatBitsToInt(_rgb) >> 16) & 0x0000ff) / 255.0; \n"
            "  float g = float((floatBitsToInt(_rgb) >> 8) & 0x0000ff) / 255.0; \n"
            "  float b = float(floatBitsToInt(_rgb) & 0x0000ff) / 255.0; \n"
            "  color = vec3(r, g, b); \n"
            "} \n "
            "";

    static const char *pointcloud_fragment_shader =
            "#version 460 \n"
            "in vec3 color; \n"
            "out vec4 color_out;"
            "void main() { \n"
            "  color_out = vec4(color, 1.0); \n"
            "} \n"
            "";

    static const char *mesh_vertex_shader =
            "#version 460 \n"
            "uniform mat4 mvp; \n"
            "in vec4 pos; \n"
            "in vec2 _uv; \n"
            "out vec2 uv; \n"
            "void main() { \n"
            "  gl_Position = mvp * pos; \n"
            "  uv = _uv; \n"
            "} \n "
            "";

    static const char *mesh_fragment_shader =
            "#version 460 \n"
            "uniform sampler2D tex; \n"
            "in vec2 uv; \n"
            "out vec4 out_color; \n"
            "void main() { \n"
            "  vec2 flipped_uv = vec2(uv.x, 1.0-uv.y);\n"
            "  out_color = texture(tex, flipped_uv);\n"
            "} \n"
            "";


    static const char *color_point_vertex_shader =
            "#version 460 \n"
            "uniform mat4 mvp; \n"
            "uniform float point_size; \n"
            "in vec4 pos; \n"
            "void main() { \n"
            "  gl_Position = mvp * pos; \n"
            "  gl_PointSize = point_size; \n"
            "} \n "
            "";

    static const char *color_point_fragment_shader =
            "#version 460 \n"
            "uniform vec3 color; \n"
            "out vec4 color_out; \n"
            "void main() { \n"
            "  color_out = vec4(color, 1.0); \n"
            "} \n"
            "";

    using Frame = struct Frame {
        ColorMesh mesh;
        CloudConstPtrT cloud;
        ImagePtrT image;
        std::vector<float> scanLandmarks;
        std::vector<float> meshLandmarks;
    };

    Frame getFrame(telef::vis::FittingVisualizer::InputPtrT input) {
        auto model = input->pca_model;
        auto mesh = model->genMesh(input->shapeCoeff, input->expressionCoeff);
        auto image = input->image;
        mesh.applyTransform(input->transformation);
        projectColor(image, mesh, input->fx, input->fy);
        auto cloud = input->cloud;

        const size_t lmkCount = input->pca_model->getLandmarks().size();
        std::vector<float> meshLandmarks(lmkCount*3);
        for(size_t i=0; i<lmkCount; i++) {
            std::copy_n(mesh.position.data() + 3*(input->pca_model->getLandmarks()[i]), 3, &meshLandmarks[3*i]);
        }

        std::vector<float> scanLandmarks(lmkCount*3);
        for(size_t i=0; i<lmkCount; i++) {
            scanLandmarks[3*i+0] = input->landmark3d->points[i].x;
            scanLandmarks[3*i+1] = input->landmark3d->points[i].y;
            scanLandmarks[3*i+2] = input->landmark3d->points[i].z;
        }

        return Frame {.mesh=mesh, .cloud=cloud, .image=image,
                .scanLandmarks=scanLandmarks,
                .meshLandmarks=meshLandmarks};
    }

    GLuint compileShader(const char *source, GLenum type) {
        GLuint result = glCreateShader(type);
        glShaderSource(result, 1, &source, NULL);
        glCompileShader(result);

        GLint status;
        glGetShaderiv(result, GL_COMPILE_STATUS, &status);

        if(status != GL_TRUE) {
            char buffer[512];
            glGetShaderInfoLog(result, 512, NULL, buffer);
            throw std::runtime_error(buffer);
        }

        return result;
    }

    GLuint getShaderProgram(const char *vertexShaderSource, const char* fragmentShaderSource) {
        GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        return program;
    }
}

namespace telef::vis {

    FittingVisualizer::FittingVisualizer()
            : renderRunning{true},
              renderThread(&FittingVisualizer::render, this),
              renderTarget{nullptr},
              clickInitialized{false},
              phi{M_PI},
              theta{0.0f},
              trackballMode(None),
              translation{0.0f, 0.0f, 0.8f},
              zoom{1.0f} {}

    FittingVisualizer::~FittingVisualizer() {
        renderRunning = false;
        renderThread.join();
    }

    void FittingVisualizer::process(FittingVisualizer::InputPtrT input) {
        renderMutex.lock();
        renderTarget = input;
        renderMutex.unlock();
    }

    void FittingVisualizer::stop() {
        renderRunning = false;
    }

    void FittingVisualizer::render() {
        init();
        window = getWindow(this);

        // Generate VBOs
        glGenBuffers(1, &pointCloud);
        glGenBuffers(1, &scanLandmark);
        glGenBuffers(1, &meshLandmark);
        glGenBuffers(1, &meshTriangles);
        glGenBuffers(1, &meshPosition);
        glGenTextures(1, &meshTexture);
        glGenBuffers(1, &meshUVCoords);
        glGenBuffers(1, &colorPointPosition);

        pointCloudShader = getShaderProgram(pointcloud_vertex_shader, pointcloud_fragment_shader);
        meshShader = getShaderProgram(mesh_vertex_shader, mesh_fragment_shader);
        colorPointShader = getShaderProgram(color_point_vertex_shader, color_point_fragment_shader);

        Frame frame;
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);

        while(!glfwWindowShouldClose(window) && renderRunning) {
            glfwPollEvents();

            // Get Render Target
            auto targetCurrFrame = safeGetInput();
            if(targetCurrFrame) {
                frame = getFrame(targetCurrFrame);
            }
            if(!frame.cloud) continue;

            // Start drawing
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            drawPointCloud(frame.cloud);
            drawMesh(frame.mesh, frame.image);
            drawColorPoints(frame.meshLandmarks, 10.0f, 1.0f, 0.0f, 0.0f);
            drawColorPoints(frame.scanLandmarks, 10.0f, 0.0f, 0.0f, 1.0f);

            glfwSwapBuffers(window);
        }

        glfwTerminate();
    }

    void FittingVisualizer::drawPointCloud(CloudConstPtrT cloud) {
        Eigen::Matrix4f mvp = getMvpMatrix();

        // Draw point cloud scan
        glUseProgram(pointCloudShader);
        GLint mvpPosition = glGetUniformLocation(pointCloudShader, "mvp");
        glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());

        glEnableVertexAttribArray(0); //pos
        glEnableVertexAttribArray(1); //_rgb
        glBindBuffer(GL_ARRAY_BUFFER, pointCloud);
        glBufferData(GL_ARRAY_BUFFER,
                     cloud->points.size()*sizeof(pcl::PointXYZRGBA),
                     cloud->points.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              sizeof(pcl::PointXYZRGBA), reinterpret_cast<void*>(offsetof(pcl::PointXYZRGBA, x)));
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                              sizeof(pcl::PointXYZRGBA), reinterpret_cast<void*>(offsetof(pcl::PointXYZRGBA, rgba)));
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(cloud->points.size()));
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
    }

    void FittingVisualizer::drawMesh(const ColorMesh &mesh, ImagePtrT image) {
        std::vector<unsigned int> triangles(mesh.triangles.size()*3);
        for(int i=0; i<mesh.triangles.size(); i++) {
            std::copy_n(mesh.triangles[i].data(), 3, &triangles[3*i]);
        }
        Eigen::Matrix4f mvp = getMvpMatrix();

        glUseProgram(meshShader);
        GLint mvpPosition = glGetUniformLocation(meshShader, "mvp");
        glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());
        glEnableVertexAttribArray(0); //pos
        glEnableVertexAttribArray(1); //_uv
        glBindBuffer(GL_ARRAY_BUFFER, meshPosition);
        glBufferData(GL_ARRAY_BUFFER,
                     mesh.position.size()*sizeof(float),
                     mesh.position.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, meshUVCoords);
        glBufferData(GL_ARRAY_BUFFER,
                     mesh.uv.size()*sizeof(float),
                     mesh.uv.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshTriangles);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size()*sizeof(int), triangles.data(), GL_STREAM_DRAW);

        glBindTexture(GL_TEXTURE_2D, meshTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     image->getWidth(), image->getHeight(), 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     image->getData());
        GLint texPosition = glGetUniformLocation(meshShader, "tex");
        glUniform1i(texPosition, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(triangles.size()), GL_UNSIGNED_INT, NULL);
        glDisableVertexAttribArray(0); //pos
        glDisableVertexAttribArray(1); //_uv
    }

    void FittingVisualizer::drawColorPoints(const std::vector<float> &points,
                                            float pointSize, float r, float g, float b) {
        Eigen::Matrix4f mvp = getMvpMatrix();

        glUseProgram(colorPointShader);
        GLint mvpPosition = glGetUniformLocation(colorPointShader, "mvp");
        glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());
        glUniform1f(glGetUniformLocation(colorPointShader, "point_size"), pointSize);
        glUniform3f(glGetUniformLocation(colorPointShader, "color"), r, g, b);

        glEnableVertexAttribArray(0); //pos
        glBindBuffer(GL_ARRAY_BUFFER, colorPointPosition);
        glBufferData(GL_ARRAY_BUFFER,
                     points.size()*sizeof(float),
                     points.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));
        glDisableVertexAttribArray(0);
    }

    FittingVisualizer::InputPtrT FittingVisualizer::safeGetInput() {
        renderMutex.lock();
        if(!renderTarget) {
            renderMutex.unlock();
            return nullptr;
        }

        auto targetCurrFrame = renderTarget;
        renderTarget = nullptr;
        renderMutex.unlock();
        return targetCurrFrame;
    }

    Eigen::Matrix4f FittingVisualizer::getMvpMatrix() {
        Eigen::Vector3f up;
        up << 0.0f, 1.0f, 0.0f;
        Eigen::Vector3f side;
        side << sinf(theta), 0.0f, cosf(theta);
        Eigen::Vector3f rotationAxis = up.cross(side);

        Eigen::Matrix4f view =
                (Eigen::AngleAxis<float>(phi, rotationAxis)
                 * Eigen::Translation3f(translation[0], translation[1], translation[2])).matrix();

        const float zFar = 1024.0f;
        const float zNear = 1.0f;
        Eigen::Matrix4f proj;
        auto yscale = 1.0f/tanf((M_PI*zoom*0.5) / 2);
        auto xscale = yscale/(16.0f/9.0f);
        proj << xscale, 0, 0, 0,
                0, yscale, 0, 0,
                0, 0, -zFar/(zFar-zNear), -1,
                0, 0, -zNear*zFar/(zFar-zNear), 0;

        return proj * view;
    }


    void FittingVisualizer::mousePositionCallback(GLFWwindow *window, double xpos, double ypos) {
        auto mode = trackballMode;

        if(mode != FittingVisualizer::None) {
            if(!clickInitialized) {
                clickXPos = xpos;
                clickYPos = ypos;
                clickPhi = phi;
                clickTheta = theta;
                std::copy_n(translation, 3, clickTranslation);
                clickInitialized = true;
            }

            double ox = clickXPos;
            double oy = clickYPos;

            if(mode == FittingVisualizer::Rotating) {
                float thetaDelta = static_cast<float>((xpos - ox) * 0.001);
                float phiDelta = static_cast<float>((ypos - oy) * 0.001);
                this->theta = clickTheta + thetaDelta;
                this->phi = clickPhi + phiDelta;
            }
            else if(mode == FittingVisualizer::Panning) {
                float x = static_cast<float>((xpos - ox) * 0.01);
                float y = static_cast<float>((ypos - oy) * 0.01);
                translation[0] = clickTranslation[0] + x;
                translation[1] = clickTranslation[1] + y;
            }
        }
    }

    void FittingVisualizer::mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
        auto zoom = this->zoom;
        this->zoom = static_cast<float>(zoom + 0.01*yoffset);

    }

    void FittingVisualizer::mouseButtonCallbackGLFW(GLFWwindow *window, int button, int action, int mods) {
        if(action == GLFW_PRESS) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_LEFT:
                    trackballMode = FittingVisualizer::Rotating;
                    clickInitialized = false;
                    break;

                case GLFW_MOUSE_BUTTON_MIDDLE:
                    trackballMode = FittingVisualizer::Panning;
                    clickInitialized = false;
                    break;
            }
        }
        if(action == GLFW_RELEASE) {
            trackballMode = FittingVisualizer::None;
        }
    }
}
