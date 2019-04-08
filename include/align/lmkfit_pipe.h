#pragma once

#include "align/nonrigid_pipe.h"

namespace
{
using namespace telef::io;
}

namespace telef::align
{
/**
 * Fit shape and expression coefficients using linear equations
 */
class LmkFitPipe :
    public Pipe<PCANonRigidAlignmentSuite, PCANonRigidAlignmentSuite>
{
public:
  LmkFitPipe(float reg=0.0001);
private:
  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in) override;

  float m_reg;
  Eigen::VectorXf m_prevShape;
};
}
