#pragma once

#include "align/nonrigid_pipe.h"
#include "io/pipe.h"

namespace
{
using namespace telef::io;
using namespace telef::align;
}

namespace telef::intrinsic
{
class IntrinsicPipe : public Pipe<PCANonRigidFittingResult, PCANonRigidFittingResult>
{
public:

private:
  boost::shared_ptr<PCANonRigidFittingResult>
  _processData(boost::shared_ptr<PCANonRigidFittingResult> in) override;
};
}
