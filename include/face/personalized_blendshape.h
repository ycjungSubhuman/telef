#pragma once

#include "face/mo"

namespace telef::face
{
class PersonalizedBlendshape
{
public:
  PersonalizedBlendshape(
      Eigen::VectorXf neutral,
      Eigen::MatrixXf mean_exp_delta_matrix);
);

  /**
   * Neutral face vertex positions
   */
  Eigen::VectorXf getNeutral() const;
  /**
   * Each column of this matrix is delta to each target shape
   * from neutral face
   */
  Eigen::MatrixXf getDeltaMatrix() const;

  /**
   * Number of expression basis
   */
  int getRank() const;
private:

  Eigen::VectorXf m_neutral;
  Eigen::MatrixXf m_delta;
};
}
