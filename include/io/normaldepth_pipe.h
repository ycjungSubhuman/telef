#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "io/pipe.h"
#include "align/nonrigid_pipe.h"

namespace {
using namespace telef::io;
using namespace telef::align;
} // namespace
/**
 * Saves fitting result to to multiple PNG files (RGB/Depth/Normal)
 *
 * Note)
 * This renderer does not work will varying resolution.
 * If the first frame it received is AxB, the other frames should be
 * AxB as well.
 */
namespace telef::io
{
class MeshNormalDepthRenderer
  : public telef::io::Pipe<PCANonRigidFittingResult, PCANonRigidFittingResult> {
public:
  using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;

  InputPtrT _processData(InputPtrT input) override;

private:
  void initFrameBuffers(InputPtrT input);

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
}
