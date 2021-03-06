#pragma once

#define NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS \
  (285.63f) // Based on 320x240 pixel size.
#define NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS \
  (3.501e-3f) // (1/NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS)
#define NUI_CAMERA_DEPTH_NOMINAL_DIAGONAL_FOV (70.0f)
#define NUI_CAMERA_DEPTH_NOMINAL_HORIZONTAL_FOV (58.5f)
#define NUI_CAMERA_DEPTH_NOMINAL_VERTICAL_FOV (45.6f)

#define NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS \
  (531.15f) // Based on 640x480 pixel size.
#define NUI_CAMERA_COLOR_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS \
  (1.83e-3f) // (1/NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS)
#define NUI_CAMERA_COLOR_NOMINAL_DIAGONAL_FOV (73.9f)
#define NUI_CAMERA_COLOR_NOMINAL_HORIZONTAL_FOV (62.0f)
#define NUI_CAMERA_COLOR_NOMINAL_VERTICAL_FOV (48.6f)
