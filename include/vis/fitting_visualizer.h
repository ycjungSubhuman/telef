#pragma once

#include "align/nonrigid_pipe.h"
#include "io/frontend.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <mutex>
#include <thread>

namespace {
using namespace telef::io;
using namespace telef::align;
} // namespace

namespace telef::vis {
class FittingVisualizer : public telef::io::FrontEnd<PCANonRigidFittingResult> {
public:
  FittingVisualizer(const int geoMaxPoints = 2000,
		    const float geoSearchRadius = 0.005);
  ~FittingVisualizer();
  using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;
  void process(InputPtrT input) override;
  void stop() override;
  void mousePositionCallback(GLFWwindow *window, double xpos, double ypos);
  void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
  void mouseButtonCallback(GLFWwindow *window, int button, int action,
			   int mods);
  void keyCallback(GLFWwindow *window, int key, int scancode, int action,
		   int mods);

private:
  void render();
  InputPtrT safeGetInput();
  Eigen::Matrix4f getMvpMatrix();

  void drawPointCloud(CloudConstPtrT cloud);
  void drawMesh(const ColorMesh &mesh, const std::vector<float> &normal,
		ImagePtrT image);
  void drawColorPoints(const std::vector<float> &points, float pointSize,
		       float r, float g, float b);
  void drawCorrespondence(const std::vector<float> &pointSet1,
			  const std::vector<float> &pointSet2, float r, float g,
			  float b);
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

  // 0: Color, 1: No Color, 2: wireframe
  static constexpr int meshModeCount = 3;
  int meshMode;

  // Geometric Term Visualization
  int geoMaxPoints;
  float geoSearchRadius;
};

/**
 * Saves fitting result to to multiple PNG files (RGB/Depth/Normal)
 * 
 * Note)
 * This renderer does not work will varying resolution.
 * If the first frame it received is AxB, the other frames should be
 * AxB as well.
 */
class MeshNormalDepthRenderer
    : public telef::io::AsyncFrontEnd<PCANonRigidFittingResult> {
public:
  using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;

  /**
   * @param record_root 	recoding root folder. All the rendered images will
   *				be saved under this folder
   * @param filename_generator	takes frame index, returns filename without extension.
   *				If not given, defaults to std::to_string
   * @param color_ext		extension for color image file.
   * 				If not given, defaults to .png
   * @param depth_ext		extension for depth image file.
   * 				If not given, defaults to .d.png
   * @param normal_ext		extension for normal image file.
   * 				If not given, defaults to .n.png
   */
  MeshNormalDepthRenderer(fs::path record_root,
                          std::function<std::string(int i)> filename_generator=[](int i){return std::to_string(i);},
                          std::string color_ext=".png", std::string depth_ext=".d.png",
                          std::string normal_ext=".n.png");

  virtual ~MeshNormalDepthRenderer();

  void _process(InputPtrT input) override;

private:

  void initFrameBuffers(InputPtrT input);

  fs::path m_record_root;
  std::function<std::string(int i)> m_filename_generator;
  std::string m_color_ext;
  std::string m_depth_ext;
  std::string m_normal_ext;

  GLuint m_normal_prog;
  GLuint m_fb; // framebuffer
  GLuint m_rb; // renderbufer
  GLuint m_drb; // depth renderbuffer

  GLuint m_vbuf;
  GLuint m_nbuf;
  GLuint m_tbuf;

  int m_maybe_width;
  int m_maybe_height;

  int m_index;

  GLFWwindow *m_dummy_window;
};
} // namespace telef::vis
