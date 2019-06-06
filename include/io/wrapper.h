#pragma once

#include "io/pipe.h"
#include "align/nonrigid_pipe.h"
#include "io/normaldepth_pipe.h"


namespace telef::io
{
class MeshNormalDepthRendererWrapper
  : public telef::io::Pipe<PCANonRigidFittingResult, PCANonRigidFittingResult> {
public:
  using InputPtrT = boost::shared_ptr<PCANonRigidFittingResult>;

  MeshNormalDepthRendererWrapper();

  InputPtrT _processData(InputPtrT input) override;

private:
    boost::shared_ptr<MeshNormalDepthRenderer> inner;
  };
}