#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstddef>
#include <Eigen/Core>
#define _USE_MATH_DEFINES
#include <cmath>

#include "vis/fitting_visualizer.h"
#include "mesh/colormapping.h"
#include "util/cudautil.h"
#include "face/cu_model_kernel.h"
#include "face/model_cudahelper.h"
#include "face/raw_model.h"
#include "io/ply/meshio.h"

#include <cv.h>
#include <highgui.h>

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
        visualizer->mouseButtonCallback(window, button, action, mods);
    }

    void mousePositionCallbackGLFW(GLFWwindow *window, double xpos, double ypos) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->mousePositionCallback(window, xpos, ypos);
    }

    void mouseScrollCallbackGLFW(GLFWwindow *window, double xoffset, double yoffset) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->mouseScrollCallback(window, xoffset, yoffset);
    }

    void keyCallbackGLFW(GLFWwindow *window, int key, int scancode, int action, int mods) {
        FittingVisualizer *visualizer = static_cast<FittingVisualizer *>(glfwGetWindowUserPointer(window));
        visualizer->keyCallback(window, key, scancode, action, mods);
    }

    GLFWwindow *getWindow (FittingVisualizer *visualizer) {
//        auto window = glfwCreateWindow(1920, 1080, "Fitting Visualizer", NULL, NULL);
        auto window = glfwCreateWindow(640, 480, "Fitting Visualizer", NULL, NULL);
        if(!window) {
            killGlfw("Error on GLFW window creation");
        }
        glfwSetWindowUserPointer(window, visualizer);

        // Register input callbacks
        glfwSetMouseButtonCallback(window, mouseButtonCallbackGLFW);
        glfwSetCursorPosCallback(window, mousePositionCallbackGLFW);
        glfwSetScrollCallback(window, mouseScrollCallbackGLFW);
        glfwSetKeyCallback(window, keyCallbackGLFW);

        glfwMakeContextCurrent(window);
        if(glewInit() != GLEW_OK) {
            throw std::runtime_error("GLEW init fail");
        }
        glfwSwapInterval(1);
        return window;
    }

    const char *pointcloud_vertex_shader =
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

    const char *pointcloud_fragment_shader =
            "#version 460 \n"
            "in vec3 color; \n"
            "out vec4 color_out;"
            "void main() { \n"
            "  color_out = vec4(color, 1.0); \n"
            "} \n"
            "";

    const char *mesh_vertex_shader =
            "#version 460 \n"
            "uniform mat4 mvp; \n"
            "in vec4 pos; \n"
            "in vec2 _uv; \n"
            "in vec3 _normal; \n"
            "out vec2 uv; \n"
            "out vec3 normal; \n"
            "void main() { \n"
            "  gl_Position = mvp * pos; \n"
            "  uv = _uv; \n"
            "  normal = normalize(_normal); \n"
            "} \n "
            "";

    const char *mesh_fragment_shader =
            "#version 460 \n"
            "uniform sampler2D tex; \n"
            "uniform int mesh_mode; \n"
            "in vec2 uv; \n"
            "in vec3 normal; \n"
            "out vec4 out_color; \n"
            "void main() { \n"
            "  const vec2 flipped_uv = vec2(uv.x, 1.0-uv.y);\n"
            "  const vec3 light = normalize(vec3(0.0, 0.0, 1.0)); \n"
            "  const float intensity = clamp(dot(light, normalize(normal)), 0.0, 1.0); \n"
            "  if(0 == mesh_mode) { \n"
            "    out_color = intensity * texture(tex, flipped_uv);\n"
            "  } else { \n"
            "    out_color = vec4(intensity * vec3(1.0, 1.0, 1.0), 1.0);\n"
            "  } \n"
            "} \n"
            "";

    const char *normal_fragment_shader =
            "#version 460 \n"
            "uniform sampler2D tex; \n"
            "uniform int mesh_mode; \n"
            "in vec2 uv; \n"
            "in vec3 normal; \n"
            "out vec4 out_color; \n"
            "void main() { \n"
            "  if (abs(normal.x)+abs(normal.y)+abs(normal.z) > 0) {"
            "     out_color = vec4(normal, 1.0);\n"
            "  } else {"
            "     out_color = vec4(0.0,0.0,0.0,0.0);"
            "  }"
            "} \n"
            "";

    const char *point_vertex_shader =
            "#version 460 \n"
            "uniform mat4 mvp; \n"
            "uniform float point_size; \n"
            "in vec4 pos; \n"
            "void main() { \n"
            "  gl_Position = mvp * pos; \n"
            "  gl_PointSize = point_size; \n"
            "} \n "
            "";

    const char *color_fragment_shader =
            "#version 460 \n"
            "uniform vec3 color; \n"
            "out vec4 color_out; \n"
            "void main() { \n"
            "  color_out = vec4(color, 1.0); \n"
            "} \n"
            "";

    using Frame = struct Frame {
        ColorMesh mesh;
        std::vector<float> vertexNormal;
        CloudConstPtrT cloud;
        ImagePtrT image;
        std::vector<float> scanLandmarks;
        std::vector<float> meshLandmarks;
        std::vector<float> scanGeo;
        std::vector<float> meshGeo;
        float fx;
        float fy;
    };

    Frame getFrame(telef::vis::FittingVisualizer::InputPtrT input, const int geoMaxPoints, const float geoRadius) {
        auto model = input->pca_model;
        auto mesh = model->genMesh(input->shapeCoeff, input->expressionCoeff);


        mesh.vertexNormals = std::vector<float>(static_cast<unsigned long>(mesh.position.size()), 0.0f);
        for(int i=0; i<mesh.triangles.size(); i++) {
            const auto v1Ind = mesh.triangles[i][0];
            const auto v2Ind = mesh.triangles[i][1];
            const auto v3Ind = mesh.triangles[i][2];
            Eigen::Vector3f v1 = mesh.position.segment(3*v1Ind, 3);
            Eigen::Vector3f v2 = mesh.position.segment(3*v2Ind, 3);
            Eigen::Vector3f v3 = mesh.position.segment(3*v3Ind, 3);
            Eigen::Vector3f unnormalizedNormal = (v2 - v1).cross(v3 - v1).normalized();
            mesh.vertexNormals[3*v1Ind + 0] += unnormalizedNormal[0];
            mesh.vertexNormals[3*v1Ind + 1] += unnormalizedNormal[1];
            mesh.vertexNormals[3*v1Ind + 2] += unnormalizedNormal[2];

            mesh.vertexNormals[3*v2Ind + 0] += unnormalizedNormal[0];
            mesh.vertexNormals[3*v2Ind + 1] += unnormalizedNormal[1];
            mesh.vertexNormals[3*v2Ind + 2] += unnormalizedNormal[2];

            mesh.vertexNormals[3*v3Ind + 0] += unnormalizedNormal[0];
            mesh.vertexNormals[3*v3Ind + 1] += unnormalizedNormal[1];
            mesh.vertexNormals[3*v3Ind + 2] += unnormalizedNormal[2];
        }

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

        pcl::PointCloud<PointT>::Ptr emptyLmks (new pcl::PointCloud<PointT>);
        std::vector<int> scanLmkIdx;
        std::vector<int> meshGeoIdx;
        std::vector<int> scanGeoIdx;
        int numCorr = 0;

        int nMeshPoints = mesh.position.rows()/3;
        int nMeshSize = mesh.position.rows();


        //Device
        int* meshCorr_d;
        int* scanCorr_d;
        float* distance_d;
        int* numCorr_d;

        float* mesh_d;
        CUDA_CHECK(cudaMalloc((void**)(&meshCorr_d), geoMaxPoints*sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)(&scanCorr_d), geoMaxPoints*sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)(&distance_d), geoMaxPoints*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)(&numCorr_d), sizeof(int)));

        CUDA_CHECK(cudaMalloc((void**)(&mesh_d), nMeshSize*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(mesh_d,mesh.position.data(), nMeshSize*sizeof(float), cudaMemcpyHostToDevice));

        C_ScanPointCloud scan;
        loadScanToCUDADevice(&scan, input->cloud, input->fx, input->fy, scanLmkIdx, input->transformation, emptyLmks);

        find_mesh_to_scan_corr(meshCorr_d, scanCorr_d, distance_d, numCorr_d, mesh_d, nMeshSize, scan,
                               geoRadius, geoMaxPoints);

        CUDA_CHECK(cudaMemcpy(&numCorr, numCorr_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (numCorr > geoMaxPoints){
            numCorr = geoMaxPoints;
            cout << "Corrected NumCorr:" << numCorr << endl;
        }
        meshGeoIdx.resize(numCorr);
        scanGeoIdx.resize(numCorr);

        CUDA_CHECK(cudaMemcpy(meshGeoIdx.data(), meshCorr_d, numCorr * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scanGeoIdx.data(), scanCorr_d, numCorr * sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(meshCorr_d));
        CUDA_CHECK(cudaFree(scanCorr_d));
        CUDA_CHECK(cudaFree(distance_d));
        CUDA_CHECK(cudaFree(mesh_d));
        freeScanCUDA(scan);

        std::vector<float> meshGeo, scanGeo;
        for (int idx = 0; idx < numCorr; idx++)
        {
            auto scanPnt = input->cloud->at(scanGeoIdx[idx]);
            scanGeo.push_back(scanPnt.x);
            scanGeo.push_back(scanPnt.y);
            scanGeo.push_back(scanPnt.z);

            meshGeo.push_back(mesh.position(3*meshGeoIdx[idx]));
            meshGeo.push_back(mesh.position(3*meshGeoIdx[idx]+1));
            meshGeo.push_back(mesh.position(3*meshGeoIdx[idx]+2));
        }

        //telef::io::ply::writeObjMesh("test.obj", mesh);

        return Frame {.mesh=mesh, .vertexNormal=mesh.vertexNormals, .cloud=cloud, .image=image,
                .scanLandmarks=scanLandmarks,
                .meshLandmarks=meshLandmarks,
                .scanGeo=scanGeo, .meshGeo=meshGeo,
                .fx=input->fx, .fy = input->fy};
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

    FittingVisualizer::FittingVisualizer(const int geoMaxPoints, const float geoSearchRadius)
            : renderRunning{true},
              renderThread(&FittingVisualizer::render, this),
              renderTarget{nullptr},
              clickInitialized{false},
              phi{M_PI},
              theta{0.0f},
              trackballMode(None),
              translation{0.0f, 0.0f, 0.0f},
              zoom{1.0f},
              meshMode(0),
              geoMaxPoints(geoMaxPoints),
              geoSearchRadius(geoSearchRadius){}

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
        glGenBuffers(1, &lineCorrespondence);
        glGenBuffers(1, &meshNormal);

        pointCloudShader = getShaderProgram(pointcloud_vertex_shader, pointcloud_fragment_shader);
//        meshShader = getShaderProgram(mesh_vertex_shader, mesh_fragment_shader);
        meshShader = getShaderProgram(mesh_vertex_shader, normal_fragment_shader);
        colorPointShader = getShaderProgram(point_vertex_shader, color_fragment_shader);

        Frame frame;
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);

        while(!glfwWindowShouldClose(window) && renderRunning) {
            glfwPollEvents();

            // Get Render Target
            auto targetCurrFrame = safeGetInput();
            if(targetCurrFrame) {
                frame = getFrame(targetCurrFrame, geoMaxPoints, geoSearchRadius);

            }
            if(!frame.cloud) continue;

            fx = frame.fx;
            fy = frame.fy;

            // Start drawing
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            auto width = frame.image->getWidth();
            auto height = frame.image->getHeight();


            //glViewport(0, 0, 960, 540);
//            glViewport(0, 0, width, height);
//            drawPointCloud(frame.cloud);
//            drawMesh(frame.mesh, frame.vertexNormal, frame.image);
//            drawColorPoints(frame.meshLandmarks, 10.0f, 1.0f, 0.0f, 0.0f);
//            drawColorPoints(frame.scanLandmarks, 10.0f, 0.0f, 0.0f, 1.0f);
//            drawCorrespondence(frame.meshLandmarks, frame.scanLandmarks, 0.0f, 1.0f, 0.0f);
//            drawColorPoints(frame.meshGeo, 5.0f, 1.0f, 1.0f, 1.0f);
//            drawColorPoints(frame.scanGeo, 5.0f, 0.0f, 0.3f, 0.5f);
//            drawCorrespondence(frame.meshGeo, frame.scanGeo, 0.5f, 0.5f, 0.5f);


            //glViewport(0, 540, 960, 540);
            glViewport(0, 0, width, height);
            drawMesh(frame.mesh, frame.vertexNormal, frame.image);

            //glViewport(960, 0, 960, 540);
//            glViewport(width, 0, width, height);
//            drawColorPoints(frame.meshLandmarks, 10.0f, 1.0f, 0.0f, 0.0f);
//            drawColorPoints(frame.scanLandmarks, 10.0f, 0.0f, 0.0f, 1.0f);
//            drawCorrespondence(frame.meshLandmarks, frame.scanLandmarks, 0.0f, 1.0f, 0.0f);
//            drawColorPoints(frame.meshGeo, 5.0f, 1.0f, 1.0f, 1.0f);
//            drawColorPoints(frame.scanGeo, 5.0f, 0.0f, 0.3f, 0.5f);
//            drawCorrespondence(frame.meshGeo, frame.scanGeo, 0.5f, 0.5f, 0.5f);

            //glViewport(960, 540, 960, 540);
//            glViewport(width, height, width, height);
//            drawPointCloud(frame.cloud);
//            drawColorPoints(frame.scanLandmarks, 10.0f, 0.0f, 0.0f, 1.0f);


            glfwSwapBuffers(window);

            std::vector< unsigned char > pixels( width * height * 4 );
            glReadPixels(0, 0, width, height,
                         GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

//            static int nframe = 0;

            auto img = cv::Mat(height, width, CV_8UC4);
            memcpy(img.data, pixels.data(), pixels.size()*sizeof(uchar));

            cvtColor(img, img, CV_RGBA2BGRA);
            cv::Mat flipped;
            cv::flip(img, flipped , 0);
//            cv::imwrite("test_normals" + std::to_string(nframe) + ".bmp", flipped);
            cv::imwrite("test_normals" + std::to_string(0) + ".png", flipped);
//            telef::io::saveBMPFile("normals_frame" + std::to_string(nframe) + ".bmp",
//                    pixels.data(), frame.image->getWidth(), frame.image->getHeight());
//            nframe++;

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

    void FittingVisualizer::drawMesh(const ColorMesh &mesh, const std::vector<float> &normal, ImagePtrT image) {
        std::vector<unsigned int> triangles(mesh.triangles.size()*3);
        for(int i=0; i<mesh.triangles.size(); i++) {
            std::copy_n(mesh.triangles[i].data(), 3, &triangles[3*i]);
        }
        Eigen::Matrix4f mvp = getMvpMatrix(); // * Eigen::Affine3f(Eigen::Scaling(2.f, 2.f, 2.f)).matrix();

        glUseProgram(meshShader);
        GLint mvpPosition = glGetUniformLocation(meshShader, "mvp");
        glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());
        glUniform1i(glGetUniformLocation(meshShader, "mesh_mode"), meshMode);
        glEnableVertexAttribArray(0); //pos
        glEnableVertexAttribArray(1); //_uv
        glEnableVertexAttribArray(2); //_normal
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
        glBindBuffer(GL_ARRAY_BUFFER, meshNormal);
        glBufferData(GL_ARRAY_BUFFER,
                     normal.size()*sizeof(float),
                     normal.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
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

        if(meshMode != 2) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }

        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(triangles.size()), GL_UNSIGNED_INT, NULL);
        glDisableVertexAttribArray(0); //pos
        glDisableVertexAttribArray(1); //_uv
        glDisableVertexAttribArray(2); //_normal
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
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()/3));
        glDisableVertexAttribArray(0);
    }

    void FittingVisualizer::drawCorrespondence(const std::vector<float> &pointSet1, const std::vector<float> &pointSet2,
                                               float r, float g, float b) {
        Eigen::Matrix4f mvp = getMvpMatrix();

        glUseProgram(colorPointShader);
        GLint mvpPosition = glGetUniformLocation(colorPointShader, "mvp");
        glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());
        glUniform3f(glGetUniformLocation(colorPointShader, "color"), r, g, b);

        std::vector<float> corrLines;
        corrLines.resize(pointSet1.size()*2);
        assert(pointSet1.size() == pointSet2.size());

        for(int i=0; i<pointSet1.size()/3; i++) {
            std::copy_n(pointSet1.data()+(3*i), 3, corrLines.data()+(6*i));
            std::copy_n(pointSet2.data()+(3*i), 3, corrLines.data()+(6*i+3));
        }

        glEnableVertexAttribArray(0); //pos
        glBindBuffer(GL_ARRAY_BUFFER, lineCorrespondence);
        glBufferData(GL_ARRAY_BUFFER,
                     corrLines.size()*sizeof(float),
                     corrLines.data(),
                     GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(corrLines.size()/3));
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
                        * Eigen::Translation3f(0.f, 0.f, 0.f)).matrix();
                 // * Eigen::Translation3f(translation[0], translation[1], translation[2])).matrix();

        const float zFar = 10.0f;
        const float zNear = 1.1f;
        Eigen::Matrix4f proj;
        auto yscale = 1.0f/tanf((M_PI*zoom*0.5) / 2);
        auto xscale = yscale/(16.0f/9.0f);
//        proj << xscale, 0, 0, 0,
//                0, yscale, 0, 0,
//                0, 0, -zFar/(zFar-zNear), -1,
//                0, 0, -zNear*zFar/(zFar-zNear), 0;


        float cx = (static_cast<float>(640 - 1.f)) / 2.f;
        float cy = (static_cast<float>(480 - 1.f)) / 2.f;
//        float cx = 640/2;
//        float cy = 480/2;

        proj << 2.5*2*fx/640,       0,              0,                          0,
                0,              2.5*2*fy/480,       0,                          0,
                (640-2*cx)/640, (2*cy-480)/480, -(zFar+zNear)/(zFar-zNear), -1,
                0,              0,              -2*zNear*zFar/(zFar-zNear), 0;

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

    void FittingVisualizer::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
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

    void FittingVisualizer::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        if(action != GLFW_PRESS) {
            return;
        }

        if(key == GLFW_KEY_2) {
            cycleMeshMode();
        }

        if(key == GLFW_KEY_R) {
            resetCamera();
        }
    }

    void FittingVisualizer::cycleMeshMode() {
        meshMode = (meshMode+1) % meshModeCount;
    }

    void FittingVisualizer::resetCamera() {
        phi = static_cast<float>(M_PI);
        theta = 0.0f;
        translation[0] = 0.0f;
        translation[1] = 0.0f;
        translation[2] = 0.0f;
        zoom = 1.0f;
    }
}
