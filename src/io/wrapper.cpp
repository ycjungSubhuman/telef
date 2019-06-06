#include "io/wrapper.h"

namespace telef::io
{
  MeshNormalDepthRendererWrapper::MeshNormalDepthRendererWrapper()
  {
      inner = boost::make_shared<MeshNormalDepthRenderer>();
  }

  MeshNormalDepthRendererWrapper::InputPtrT 
  MeshNormalDepthRendererWrapper::_processData(MeshNormalDepthRendererWrapper::InputPtrT input)
  {
      return inner->operator()(input);
  }

}