#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <iostream>
#include <chrono>

#include "align/lmkfit_pipe.h"
#include "align/rigid_pipe.h"

namespace
{
using namespace telef::io;
using namespace telef::align;

Eigen::MatrixXf select_rows(
    const Eigen::MatrixXf &mat,
    const std::vector<int> &inds,
    const int rows,
    const Eigen::Matrix4f transformation)
{
  Eigen::MatrixXf result(rows*inds.size(), mat.cols());
  for (size_t i=0; i<inds.size(); i++)
    {
      Eigen::Matrix3Xf pos = mat.block(3*inds[i],0,3,mat.cols());
      Eigen::Matrix3Xf tpos =
        (transformation * pos.colwise().homogeneous()).colwise().hnormalized();
      result.block(rows*i,0,rows,mat.cols()) = tpos.block(0,0,rows,mat.cols());
    }
  return result;
}

boost::shared_ptr<PCANonRigidAlignmentSuite>
step_position(boost::shared_ptr<PCANonRigidAlignmentSuite> in, float reg)
{
  const auto landmark3d = in->fittingSuite->landmark3d;
  Eigen::VectorXf ref = in->pca_model->getReferenceVector();
  const auto lmkInds = in->pca_model->getLandmarks();
  const auto shapeRank = in->pca_model->getShapeRank();
  const auto expRank = in->pca_model->getExpressionRank();

  Eigen::MatrixXf shapeMat = in->pca_model->getShapeBasisMatrix();
  Eigen::MatrixXf expMat = in->pca_model->getExpressionBasisMatrix();

  Eigen::MatrixXf shapeLmkMat = select_rows(shapeMat, lmkInds, 3, in->transformation);
  Eigen::MatrixXf expLmkMat = select_rows(expMat, lmkInds, 3, in->transformation);
  Eigen::MatrixXf expLmkMat2d = select_rows(expMat, lmkInds, 2, in->transformation);

  Eigen::MatrixXf regMat =
    reg*Eigen::MatrixXf::Identity(shapeRank+expRank, shapeRank+expRank);

  // Set up RHS
  Eigen::VectorXf b = Eigen::VectorXf::Zero(3*lmkInds.size());
  for(int i=0; i<landmark3d->size(); i++)
    {
      Eigen::VectorXf m = ref.segment(3*lmkInds[i],3);
      Eigen::VectorXf tm = (in->transformation*m.homogeneous()).hnormalized();
      b(3*i+0) = landmark3d->points[i].x - tm(0);
      b(3*i+1) = landmark3d->points[i].y - tm(1);
      b(3*i+2) = landmark3d->points[i].z - tm(2);
    }

  // Set up LHS
  Eigen::MatrixXf A(3*lmkInds.size(), shapeRank+expRank);

  A.block(0,0,3*lmkInds.size(),shapeRank) = shapeLmkMat;
  A.block(0,shapeRank,3*lmkInds.size(),expRank) = expLmkMat;

  Eigen::MatrixXf ATA = A.transpose()*A + regMat;
  Eigen::VectorXf ATb = A.transpose()*b;

  Eigen::VectorXf x = ATA.householderQr().solve(ATb);

  Eigen::VectorXf idCoeff = x.segment(0,shapeRank);
  Eigen::VectorXf exCoeff = x.segment(shapeRank,expRank);

  in->shapeCoeff = idCoeff;
  in->expressionCoeff = exCoeff;

  return in;
}
}

namespace telef::align
{

LmkFitPipe::LmkFitPipe(float reg) :
    m_reg(reg),
    m_prevShape()
{}

boost::shared_ptr<PCANonRigidAlignmentSuite>
LmkFitPipe::_processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  auto res = in;
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
  if(0 == m_prevShape.size())
    {
        res = step_position(res, m_reg);
        m_prevShape = res->shapeCoeff;
    }
  else
    {
      res->shapeCoeff = m_prevShape;
    }
  std::chrono::duration<double> dur = std::chrono::system_clock::now()-start;
  std::cout << "Time (Lmkfit): " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "ms" << std::endl;
  return res;
}
}
