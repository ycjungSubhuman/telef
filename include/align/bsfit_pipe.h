#pragma once

#include "align/nonrigid_pipe.h"

namespace
{
using namespace telef::io;
}

namespace telef::align
{
/**
 * Fit blendshape
 */
class BsFitPipe :
    public Pipe<PCANonRigidAlignmentSuite, PCANonRigidAlignmentSuite>
{
public:
private:
  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in) override;
};
}
