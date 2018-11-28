#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <thread>
#include <mutex>
#include "io/frontend.h"
#include "align/nonrigid_pipe.h"

namespace {
    using namespace telef::io;
    using namespace telef::align;
}

namespace telef::vis {
    class FittingVisualizer : public telef::io::FrontEnd<PCANonRigidFittingResult> {
    public:
        FittingVisualizer(const int geoMaxPoints=2000, const float geoSearchRadius=0.005);
        ~FittingVisualizer();
        using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;
        void process(InputPtrT input) override;
        void stop() override;
        void mousePositionCallback(GLFWwindow *window, double xpos, double ypos);
        void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
        void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
        void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

    private:
        void render();
        InputPtrT safeGetInput();
        Eigen::Matrix4f getMvpMatrix();

        void drawPointCloud(CloudConstPtrT cloud);
        void drawMesh(const ColorMesh &mesh, const std::vector<float> &normal, ImagePtrT image);
        void drawColorPoints(const std::vector<float> &points, float pointSize, float r, float g, float b);
        void drawCorrespondence(const std::vector<float> &pointSet1, const std::vector<float> &pointSet2,
                                float r, float g, float b);
        void cycleMeshMode();
        void resetCamera();

        volatile bool renderRunning;
        GLFWwindow *window;
        InputPtrT renderTarget;
        std::thread renderThread;
        std::mutex renderMutex;
        float prevTime;

        GLuint pointCloud;
        GLuint scanLandmark;
        GLuint meshLandmark;
        GLuint meshTriangles;
        GLuint meshPosition;
        GLuint meshTexture;
        GLuint meshNormal;
        GLuint meshUVCoords;
        GLuint colorPointPosition;
        GLuint lineCorrespondence;

        GLuint pointCloudShader;
        GLuint meshShader;
        GLuint colorPointShader;

        enum TrackballMode {
            None,
            Rotating,
            Panning,
        };

        TrackballMode trackballMode;

        bool clickInitialized;
        double clickXPos;
        double clickYPos;
        float clickPhi;
        float clickTheta;
        float clickTranslation[3];

        float phi;
        float theta;
        float translation[3];
        float zoom;
        float fx;
        float fy;


        // 0: Color, 1: No Color, 2: wireframe
        static constexpr int meshModeCount = 3;
        int meshMode;

        // Geometric Term Visualization
        int geoMaxPoints;
        float geoSearchRadius;
    };
}