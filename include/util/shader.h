#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace telef::util
{
inline GLuint compileShader(const char *source, GLenum type) {
  GLuint result = glCreateShader(type);
  glShaderSource(result, 1, &source, NULL);
  glCompileShader(result);

  GLint status;
  glGetShaderiv(result, GL_COMPILE_STATUS, &status);

  if (status != GL_TRUE) {
    char buffer[5000];
    glGetShaderInfoLog(result, 512, NULL, buffer);
    throw std::runtime_error(buffer);
  }

  return result;
}

inline GLuint getShaderProgram(
    const char *vertexShaderSource, const char *fragmentShaderSource) {
  GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
  GLuint fragmentShader =
      compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);

  return program;
}
}
