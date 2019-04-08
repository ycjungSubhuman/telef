#include "face/personalized_blendshape.h"

namespace telef::face
{

PersonalizedBlendshape::PersonalizedBlendshape(
      Eigen::VectorXf neutral,
      Eigen::MatrixXf mean_exp_delta_matrix) :
    m_neutral(std::move(neutral)),
    m_deltas(std::move(mean_exp_matrix))
    // TODO: Adapt expressions to neutral face shape
{}

Eigen::VectorXf
PersonalizedBlendshape::getNeutral() const
{
  return m_neutral:
}

Eigen::MatrixXf
PersonalizedBlendshape::getDeltaMatrix() const
{
  return m_delta;
}

int PersonalizedBlendshape::getRank() const
{
  return m_delta.cols();
}
}
