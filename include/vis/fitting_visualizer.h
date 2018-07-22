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
        FittingVisualizer();
        ~FittingVisualizer();
        using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;
        void process(InputPtrT input) override;
        void stop() override;
        void mousePositionCallback(GLFWwindow *window, double xpos, double ypos);
        void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
        void mouseButtonCallbackGLFW(GLFWwindow *window, int button, int action, int mods);

    private:
        void render();
        InputPtrT safeGetInput();
        Eigen::Matrix4f getMvpMatrix();

        void drawPointCloud(CloudConstPtrT cloud);
        void drawMesh(const ColorMesh &mesh, ImagePtrT image);
        void drawColorPoints(const std::vector<float> &points, float pointSize, float r, float g, float b);

        volatile bool renderRunning;
        GLFWwindow *window;
        InputPtrT renderTarget;
        std::thread renderThread;
        std::mutex renderMutex;

        GLuint pointCloud;
        GLuint scanLandmark;
        GLuint meshLandmark;
        GLuint meshTriangles;
        GLuint meshPosition;
        GLuint meshTexture;
        GLuint meshUVCoords;
        GLuint colorPointPosition;

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
    };
}