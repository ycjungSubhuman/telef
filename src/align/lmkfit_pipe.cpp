#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <iostream>
#include <chrono>

#include "align/lmkfit_pipe.h"
#include "align/rigid_pipe.h"
#include "align/bsfit_pipe.h"
#include "align/rigid_pipe.h"
#include "io/ply/meshio.h"

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
  Eigen::VectorXf ref = in->pca_model->getReferenceVector() +
    (in->pca_model->getExpressionBasisMatrix()*in->expressionCoeff);
  const auto lmkInds = in->pca_model->getLandmarks();
  const auto shapeRank = in->pca_model->getShapeRank();
  Eigen::MatrixXf shapeMat = in->pca_model->getShapeBasisMatrix();
  Eigen::MatrixXf shapeLmkMat = select_rows(shapeMat, lmkInds, 3, in->transformation);

  Eigen::MatrixXf regMat =
    reg*Eigen::MatrixXf::Identity(shapeRank, shapeRank);

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
  Eigen::MatrixXf A = shapeLmkMat;

  Eigen::MatrixXf ATA = A.transpose()*A + regMat;
  Eigen::VectorXf ATb = A.transpose()*b;

  Eigen::VectorXf x = ATA.householderQr().solve(ATb);

  Eigen::VectorXf idCoeff = x;

  in->shapeCoeff = idCoeff;

  return in;
}

boost::shared_ptr<PCANonRigidAlignmentSuite>
step_shape(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  PCAGPUNonRigidFittingPipe shfit(10, 2500, 0.10, true, true, false);
  auto res = shfit(in);
  in->shapeCoeff = res->shapeCoeff;

  return in;
}

boost::shared_ptr<PCANonRigidAlignmentSuite>
step_expression(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  BsFitPipe bsfit;
  return bsfit(in);
}

boost::shared_ptr<PCANonRigidAlignmentSuite>
step_pose(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  PCARigidFittingPipe posefit;
  return posefit(in);
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
      Eigen::VectorXf shapeCoeff = res->shapeCoeff;
      for(int i=0; i<10; i++)
        {
            res = step_shape(res);
            //shapeCoeff = 0.8*shapeCoeff + 0.2*res->shapeCoeff;
            shapeCoeff = res->shapeCoeff;
            res = step_expression(res);
            res = step_pose(res);
        }
        m_prevShape = shapeCoeff;
        res->shapeCoeff = shapeCoeff;

        auto exr = in->pca_model->getExpressionRank();
        for(int i=0; i<exr; i++)
          {
            Eigen::VectorXf expressionCoeff = Eigen::VectorXf::Zero(exr);
            expressionCoeff(i) = 1.0f;
            ColorMesh bs = in->pca_model->genMesh(shapeCoeff, expressionCoeff);
            telef::io::ply::writePlyMesh("bs_"+std::to_string(i)+".ply", bs);
          }
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
