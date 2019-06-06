#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <chrono>
#include <vector>
#include "optimization.h"

#include "align/bsfit_pipe.h"
#include "io/ply/meshio.h"

namespace
{
using namespace telef::io;
using namespace telef::align;
using namespace alglib;

const std::vector<int> lmk_selection =
  {
   7, 8, 9,
   //51, 48, 54, 60, 64, 62, 57, 66,
   60, 62, 64, 66,
   37, 38, 40, 41,
   43, 44, 46, 47, 
   17, 19, 21,
   22, 24, 26,
  };

boost::shared_ptr<PCANonRigidAlignmentSuite>
step_position(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  const auto landmark3d = in->fittingSuite->landmark3d;
  Eigen::VectorXf ref = in->pca_model->getReferenceVector();
  const auto lmkInds = in->pca_model->getLandmarks();
  const auto expRank = in->pca_model->getExpressionRank();
  Eigen::MatrixXf shapeMat = in->pca_model->getShapeBasisMatrix();
  Eigen::MatrixXf expMat = in->pca_model->getExpressionBasisMatrix();
  Eigen::MatrixXf expLmkMat2d(2*lmk_selection.size(), expMat.cols());
  for (size_t i=0; i<lmk_selection.size(); i++)
    {
      Eigen::Matrix3Xf pos =
        expMat.block(3*lmkInds[lmk_selection[i]],0,3,expMat.cols());
      Eigen::Matrix3Xf tpos =
        (in->transformation * pos.colwise().homogeneous()).colwise().hnormalized();
      expLmkMat2d.block(2*i,0,2,expMat.cols()) = tpos.block(0,0,2,expMat.cols());
    }

  Eigen::VectorXf shape = ref+shapeMat*in->shapeCoeff;
  Eigen::VectorXf b2d = Eigen::VectorXf::Zero(2*lmk_selection.size());
  Eigen::VectorXf lmk = Eigen::VectorXf::Zero(3*lmk_selection.size());
  Eigen::VectorXf lmkref = Eigen::VectorXf::Zero(3*lmk_selection.size());
  for(int i=0; i<lmk_selection.size(); i++)
    {
      Eigen::VectorXf m = shape.segment(3*lmkInds[lmk_selection[i]],3);
      Eigen::VectorXf tm = (in->transformation*m.homogeneous()).hnormalized();
      b2d(2*i+0) = landmark3d->points[lmk_selection[i]].x - tm(0);
      b2d(2*i+1) = landmark3d->points[lmk_selection[i]].y - tm(1);

      lmk(3*i+0) = landmark3d->points[lmk_selection[i]].x;
      lmk(3*i+1) = landmark3d->points[lmk_selection[i]].y;
      lmk(3*i+2) = landmark3d->points[lmk_selection[i]].z;

      lmkref(3*i+0) = tm(0);
      lmkref(3*i+1) = tm(1);
      lmkref(3*i+2) = tm(2);
    }
  ColorMesh mlmk, mref;
  mlmk.position = lmk;
  mref.position = lmkref;

  Eigen::MatrixXd dExpLmkMat2d = expLmkMat2d.cast<double>();
  Eigen::VectorXd db2d = b2d.cast<double>();
  Eigen::MatrixXd Q = dExpLmkMat2d.transpose()*dExpLmkMat2d;
  Eigen::VectorXd c = -dExpLmkMat2d.transpose()*db2d;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(expRank);
  //Eigen::VectorXd bndl = -std::numeric_limits<double>::infinity()*Eigen::VectorXd::Ones(expRank);
  Eigen::VectorXd bndl = -0.05*Eigen::VectorXd::Ones(expRank);
  Eigen::VectorXd bndu = std::numeric_limits<double>::infinity()*Eigen::VectorXd::Ones(expRank);
  Eigen::VectorXd scale = Eigen::VectorXd::Ones(expRank);

  real_2d_array algQ;
  real_1d_array algc, algx0, algbndl, algbndu, algx, algscale;
  algQ.setcontent(Q.rows(), Q.cols(), Q.data());
  algc.setcontent(c.size(), c.data());
  algx0.setcontent(x0.size(), x0.data());
  algbndl.setcontent(bndl.size(), bndl.data());
  algbndu.setcontent(bndu.size(), bndu.data());
  algscale.setcontent(scale.size(), scale.data());

  minqpstate state;
  minqpcreate(expRank, state);
  minqpsetquadraticterm(state, algQ);
  minqpsetlinearterm(state, algc);
  minqpsetstartingpoint(state, algx0);
  minqpsetbc(state, algbndl, algbndu);
  minqpsetscale(state, algscale);
  minqpreport rep;
  minqpsetalgoquickqp(state, 0.0, 0.0, 0.0, 0, false);
  minqpoptimize(state);
  minqpresults(state, algx, rep);
  /*
  std::cout << int(rep.terminationtype) << " " <<
    rep.outeriterationscount <<
    " " << rep.inneriterationscount << std::endl;

  std::cout << algx.tostring(expRank) << std::endl;
  */

  Eigen::VectorXf exCoeff =
    Eigen::Map<Eigen::VectorXd>(algx.getcontent(), algx.length()).cast<float>();

  /*
  for(int i=0; i<expMat.cols(); i++)
    {
      Eigen::VectorXf res = shape+expMat.col(i);
      ColorMesh mres;
      mres.position = res;
      mres.triangles = in->pca_model->getReferenceMesh().triangles;
      telef::io::ply::writePlyMesh("cusbs_"+std::to_string(i)+".ply", mres);
    }
  */

  in->expressionCoeff = exCoeff;

  return in;
}
}

namespace telef::align
{
boost::shared_ptr<PCANonRigidAlignmentSuite>
BsFitPipe::_processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in)
{
  auto res = in;
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
  res = step_position(res);
  std::chrono::duration<double> dur = std::chrono::system_clock::now()-start;
  std::cout << "Time (BsFit): " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "ms" << std::endl;
  return res;
}
}
