#include <cassert>

#include "io/normaldepth_pipe.h"
#include "util/normal.h"
#include "util/shader.h"

namespace {
using namespace telef::io;
using namespace telef::align;
using namespace telef::util;

const char *mesh_normal_vertex_shader =
    "#version 460 \n"
    "uniform mat4 mvp; \n"
    "uniform mat4 mv; \n"
    "in vec4 pos; \n"
    "in vec3 _normal; \n"
    "out vec3 normal; \n"
    "void main() { \n"
    "  vec4 xy_z = mvp*pos; \n"
    "  vec4 _z = mv*pos; \n"
    "  xy_z /= xy_z.w; \n"
    "  _z /= _z.w; \n"
    "  vec4 xyz = vec4(xy_z.x, xy_z.y, -_z.z, 1.0); \n"
    "  gl_Position = xyz; \n"
    "  normal = _normal; \n"
    "} \n ";

const char *mesh_normal_fragment_shader =
    "#version 460 \n"
    "in vec3 normal; \n"
    "out vec4 out_color; \n"
    "void main() { \n"
    "  vec3 n_normed = normalize(normal);\n"
    "  vec3 n_posit = (0.5*n_normed)+vec3(0.5, 0.5, 0.5);\n"
    "  out_color = vec4(n_posit, 1.0);\n"
    "} \n"
    "";

auto PrepareNewContext() {
  GLFWwindow *window = glfwCreateWindow(100, 100, "", NULL, NULL);
  (nullptr != window);
  auto prev = glfwGetCurrentContext();
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK) {
    throw std::runtime_error("GLEW init fail");
  }

  return prev;
}

GLFWwindow *PrepareContext(GLFWwindow *newContext) {
  assert(nullptr != newContext);
  auto prev = glfwGetCurrentContext();
  if (prev != newContext) {
    glfwMakeContextCurrent(newContext);
    if (glewInit() != GLEW_OK) {
      throw std::runtime_error("GLEW init fail");
    }
  }

  return prev;
}

GLFWwindow *RestoreContext(GLFWwindow *prevContext) {
  auto curr = glfwGetCurrentContext();
  if (nullptr != prevContext) {
    glfwMakeContextCurrent(prevContext);
    if (glewInit() != GLEW_OK) {
      throw std::runtime_error("GLEW init fail");
    }
  }

  return curr;
}

} // namespace

namespace telef::io
{
void
MeshNormalDepthRenderer::initFrameBuffers(
    boost::shared_ptr<PCANonRigidFittingResult> input) {
  // Initialize dummy GLFW window context for framebuffer rendering
  if (!glfwInit()) {
    throw std::runtime_error("GLFW init failed");
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  GLFWwindow *window = glfwCreateWindow(100, 100, "", NULL, NULL);
  if (nullptr == window) {
    glfwTerminate();
    throw std::runtime_error("Dummy window init failed");
  }

  const auto prevWindow = PrepareNewContext();

  //     Compile normal renderer program
  m_normal_prog =
      getShaderProgram(mesh_normal_vertex_shader, mesh_normal_fragment_shader);

  glGenBuffers(1, &m_vbuf);
  glGenBuffers(1, &m_nbuf);
  glGenBuffers(1, &m_tbuf);

  m_dummy_window = RestoreContext(prevWindow);

  //    Initialize framebuffer and renderbuffer
  glGenFramebuffers(1, &m_fb);
  glGenRenderbuffers(1, &m_rb);
  glGenRenderbuffers(1, &m_drb);

  //    Setup color/depth renderbuffers and attach them to framebuffer
  glBindRenderbuffer(GL_RENDERBUFFER, m_rb);
  glRenderbufferStorage(
      GL_RENDERBUFFER,
      GL_RGBA,
      input->image->getWidth(),
      input->image->getHeight());
  glBindFramebuffer(GL_FRAMEBUFFER, m_fb);
  glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_rb);
  assert(GL_NO_ERROR == glGetError());

  glBindRenderbuffer(GL_RENDERBUFFER, m_drb);
  glRenderbufferStorage(
      GL_RENDERBUFFER,
      GL_DEPTH_COMPONENT,
      input->image->getWidth(),
      input->image->getHeight());
  glBindFramebuffer(GL_FRAMEBUFFER, m_fb);
  glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_drb);
  assert(GL_NO_ERROR == glGetError());

  assert(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER));

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_maybe_width = input->image->getWidth();
  m_maybe_height = input->image->getHeight();
  glViewport(0, 0, m_maybe_width, m_maybe_height);
}

boost::shared_ptr<PCANonRigidFittingResult>
MeshNormalDepthRenderer::_processData(boost::shared_ptr<PCANonRigidFittingResult> input) {
  // Initialize framebuffers with the first frame this renderer
  // received
  if (0 == m_maybe_width || 0 == m_maybe_height) {
    initFrameBuffers(input);
  }

  const auto prevContext = PrepareContext(m_dummy_window);

  assert(
      m_maybe_width == input->image->getWidth() &&
      m_maybe_height == input->image->getHeight());

  // Render normal image
  //    Initialize opengl render flags
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CW); // Because the model is using +Z as towards screen
  glCullFace(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //    Use normal rendering program
  glUseProgram(m_normal_prog);

  //    Setup vertex attributes
  glEnableVertexAttribArray(0); // pos
  glEnableVertexAttribArray(1); // _normal

  auto mesh =
      input->pca_model->genMesh(input->shapeCoeff, input->expressionCoeff);

  auto vertexPosition = mesh.position;
  auto vertexNormal = getVertexNormal(mesh);
  assert(vertexNormal.size() == vertexPosition.size());

  //        vertex positions
  glBindBuffer(GL_ARRAY_BUFFER, m_vbuf);
  glBufferData(
      GL_ARRAY_BUFFER,
      vertexPosition.size() * sizeof(float),
      vertexPosition.data(),
      GL_STREAM_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

  //        vertex normals
  glBindBuffer(GL_ARRAY_BUFFER, m_nbuf);
  glBufferData(
      GL_ARRAY_BUFFER,
      vertexNormal.size() * sizeof(float),
      vertexNormal.data(),
      GL_STREAM_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

  //    Setup uniforms
  Eigen::Matrix4f mv = input->transformation;
  float fx = input->fx;
  float fy = input->fy;
  float cx = m_maybe_width / 2.0f;
  float cy = m_maybe_height / 2.0f;
  float far = 1.0f;
  float near = 0.0f;

  //    traditional proejction matrix
  //    the output z component will be meaningless
  //    the shader uses raw z values (in meter)
  Eigen::Matrix4f p1;
  p1 << 
	  fx / cx, 0.0f, 0.0f, 0.0f, 
	     0.0f, fy / cy, 0.0f, 0.0f, 
	     0.0f, 0.0f, 1.0f, 0.0f, 
	     0.0f, 0.0f, -1.0f, 0.0f;
  Eigen::Matrix4f flip; // flips z coordinate to -z
  flip << 1.0, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f;

  Eigen::Matrix4f mvp = p1 * flip * mv;
  Eigen::Matrix4f _mv = flip * mv;

  Eigen::MatrixXf pos = Eigen::Map<Eigen::MatrixXf>(
      mesh.position.data(), 3, mesh.position.size() / 3);

  GLint mvpPosition = glGetUniformLocation(m_normal_prog, "mvp");
  GLint mvPosition = glGetUniformLocation(m_normal_prog, "mv");
  glUniformMatrix4fv(mvpPosition, 1, GL_FALSE, mvp.data());
  glUniformMatrix4fv(mvPosition, 1, GL_FALSE, _mv.data());

  //     Draw
  std::vector<unsigned int> triangles(mesh.triangles.size() * 3);
  for (int i = 0; i < mesh.triangles.size(); i++) {
    std::copy_n(mesh.triangles[i].data(), 3, &triangles[3 * i]);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_tbuf);
  glBufferData(
      GL_ELEMENT_ARRAY_BUFFER,
      triangles.size() * sizeof(unsigned int),
      triangles.data(),
      GL_STREAM_DRAW);
  glDrawElements(
      GL_TRIANGLES,
      static_cast<GLsizei>(triangles.size()),
      GL_UNSIGNED_INT,
      NULL);

  //    Get pixel values
  std::vector<float> raw_normals(m_maybe_width * m_maybe_height * 3);
  std::vector<uint16_t> raw_depth(m_maybe_width * m_maybe_height);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(
      0,
      0,
      m_maybe_width,
      m_maybe_height,
      GL_RGB,
      GL_FLOAT,
      raw_normals.data());
  glReadPixels(
      0,
      0,
      m_maybe_width,
      m_maybe_height,
      GL_DEPTH_COMPONENT,
      GL_UNSIGNED_SHORT,
      raw_depth.data());

  input->rendered_normal = std::move(raw_normals);
  input->rendered_depth = std::move(raw_depth);

  //    Clean up
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  RestoreContext(prevContext);

  return input;
}
}
