#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <ceres/ceres.h>
#include <cmath>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "align/cost_func.h"
#include "align/nonrigid_pipe.h"
#include "face/model_cudahelper.h"
#include "type.h"
#include "util/convert_arr.h"
#include "util/cu_quaternion.h"
#include "util/transform.h"

#define EPS 0.005

using namespace telef::types;
using namespace telef::face;

namespace telef::align {
PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe()
    : isModelInitialized(false), geoWeight(0), geoMaxPoints(0),
      geoSearchRadius(0), addGeoTerm(false) {
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Cublas could not be initialized");
  }
}

PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(
    const float geoWeight, const int geoMaxPoints, const float geoSearchRadius,
    const bool addGeoTerm, const bool usePrevFrame)
    : isModelInitialized(false), geoWeight(geoWeight),
      geoMaxPoints(geoMaxPoints), geoSearchRadius(geoSearchRadius),
      addGeoTerm(addGeoTerm), usePrevFrame(usePrevFrame) {
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Cublas could not be initialized");
  }
}

PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(
    const PCAGPUNonRigidFittingPipe &that) {
  this->isModelInitialized = false;
  this->geoWeight = that.geoWeight;
  this->geoMaxPoints = that.geoMaxPoints;
  this->geoSearchRadius = that.geoSearchRadius;
  this->addGeoTerm = that.addGeoTerm;
  this->usePrevFrame = that.usePrevFrame;
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Cublas could not be initialized");
  }
}

PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(
    PCAGPUNonRigidFittingPipe &&that) noexcept {
  this->isModelInitialized = that.isModelInitialized;
  this->c_deformModel = that.c_deformModel;
  this->cublasHandle = that.cublasHandle;
  this->geoWeight = that.geoWeight;
  this->geoMaxPoints = that.geoMaxPoints;
  this->geoSearchRadius = that.geoSearchRadius;
  this->addGeoTerm = that.addGeoTerm;
  this->usePrevFrame = that.usePrevFrame;
}

PCAGPUNonRigidFittingPipe &PCAGPUNonRigidFittingPipe::
operator=(const PCAGPUNonRigidFittingPipe &that) {
  if (&that != this) {
    if (isModelInitialized) {
      freeModelCUDA(c_deformModel);
      this->isModelInitialized = false;
    }
    cublasDestroy(cublasHandle);
    if (cublasCreate(&this->cublasHandle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Cublas could not be initialized");
    }
  }

  this->geoWeight = that.geoWeight;
  this->geoMaxPoints = that.geoMaxPoints;
  this->geoSearchRadius = that.geoSearchRadius;
  this->addGeoTerm = that.addGeoTerm;
  this->usePrevFrame = that.usePrevFrame;
  return *this;
}

PCAGPUNonRigidFittingPipe &PCAGPUNonRigidFittingPipe::
operator=(PCAGPUNonRigidFittingPipe &&that) {
  if (&that != this) {
    if (isModelInitialized) {
      freeModelCUDA(c_deformModel);
      this->isModelInitialized = false;
    }
    cublasDestroy(cublasHandle);
    this->isModelInitialized = that.isModelInitialized;
    this->c_deformModel = that.c_deformModel;
    this->cublasHandle = that.cublasHandle;
    this->geoWeight = that.geoWeight;
    this->geoMaxPoints = that.geoMaxPoints;
    this->geoSearchRadius = that.geoSearchRadius;
    this->addGeoTerm = that.addGeoTerm;
    this->usePrevFrame = that.usePrevFrame;
  }

  return *this;
}

PCAGPUNonRigidFittingPipe::~PCAGPUNonRigidFittingPipe() {
  if (isModelInitialized) {
    freeModelCUDA(c_deformModel);
  }
  cublasDestroy(cublasHandle);
}

boost::shared_ptr<PCANonRigidFittingResult>
PCAGPUNonRigidFittingPipe::_processData(
    boost::shared_ptr<PCANonRigidAlignmentSuite> in) {
  /* Load data to cuda device */

  // std::cout << "Fitting PCA model GPU" << std::endl;
  if (!isModelInitialized) {
    auto shapeBasis = in->pca_model->getShapeBasisMatrix();
    auto expressionBasis = in->pca_model->getExpressionBasisMatrix();
    auto ref = in->pca_model->getReferenceVector();
    auto meanShapeDeformation = in->pca_model->getShapeDeformationCenter();
    auto meanExpressionDeformation =
        in->pca_model->getExpressionDeformationCenter();
    auto landmarks = in->pca_model->getLandmarks();
    loadModelToCUDADevice(&this->c_deformModel, shapeBasis, expressionBasis,
                          ref, meanShapeDeformation, meanExpressionDeformation,
                          landmarks);
    isModelInitialized = true;
  }

  C_ScanPointCloud c_scanPointCloud;
  loadScanToCUDADevice(&c_scanPointCloud, in->rawCloud, in->fx, in->fy,
                       landmarkSelection, in->transformation,
                       in->fittingSuite->landmark3d);

  /* Setup Optimizer */
  // std::cout << "Fitting PCA model to scan..." << std::endl;
  auto lmkCost = new PCAGPULandmarkDistanceFunctor(
      this->c_deformModel, c_scanPointCloud, cublasHandle);
  ceres::Problem problem;

  // Reset input parameters if we are first frame or not using prev frames
  if (!usePrevFrame || shapeCoeff.size() == 0)
    shapeCoeff.assign(c_deformModel.shapeRank, 0);
  if (!usePrevFrame || expressionCoeff.size() == 0)
    expressionCoeff.assign(c_deformModel.expressionRank, 0);
  if (!usePrevFrame || t.size() == 0)
    t.assign(3, 0.0);
  if (!usePrevFrame || u.size() == 0)
    u = {3.14, 0.0, 0.0};
  problem.AddResidualBlock(lmkCost, new ceres::CauchyLoss(0.5),
                           shapeCoeff.data(), expressionCoeff.data(), t.data(),
                           u.data());

  if (addGeoTerm == true) {
    auto geoCost = new PCAGPUGeometricDistanceFunctor(
        this->c_deformModel, c_scanPointCloud, cublasHandle, geoMaxPoints * 3,
        sqrtf(geoWeight), geoSearchRadius);
    problem.AddResidualBlock(geoCost, new ceres::CauchyLoss(0.5),
                             shapeCoeff.data(), expressionCoeff.data(),
                             t.data(), u.data());
  }
  problem.AddResidualBlock(
      new L2RegularizerFunctor(c_deformModel.shapeRank, 0.0002), NULL,
      shapeCoeff.data());
  problem.AddResidualBlock(
      new LinearBarrierFunctor(c_deformModel.expressionRank, 0.0002, 10), NULL,
      expressionCoeff.data());
  problem.AddResidualBlock(new LinearUpperBarrierFunctor(
                               c_deformModel.expressionRank, 0.00002, 2, 1.0),
                           NULL, expressionCoeff.data());
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;

  /* Run Optimization */
  auto summary = ceres::Solver::Summary();
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  float fu[3];
  float ft[3];
  float r[9];
  float trans[16];
  convertArray(t.data(), ft, 3);
  convertArray(u.data(), fu, 3);
  calc_r_from_u(r, fu);
  create_trans_from_tu(trans, ft, r);
  Eigen::Map<Eigen::Matrix4f> eigenTrans(trans);

  /* Free Resources */
  freeScanCUDA(c_scanPointCloud);

  auto result = boost::make_shared<PCANonRigidFittingResult>();
  result->shapeCoeff =
      Eigen::Map<Eigen::VectorXd>(shapeCoeff.data(), c_deformModel.shapeRank)
          .cast<float>();
  result->expressionCoeff =
      Eigen::Map<Eigen::VectorXd>(expressionCoeff.data(),
                                  c_deformModel.expressionRank)
          .cast<float>();

  std::cout << "Fitted(Shape): " << std::endl;
  std::cout << result->shapeCoeff << std::endl;
  std::cout << "Fitted(Expression): " << std::endl;
  std::cout << result->expressionCoeff << std::endl;
  result->image = in->image;
  result->pca_model = in->pca_model;
  result->cloud = in->rawCloud;
  result->landmark3d = in->fittingSuite->landmark3d;
  result->fx = in->fx;
  result->fy = in->fy;
  result->transformation = eigenTrans * in->transformation;

  return result;
}

const std::vector<int> PCAGPUNonRigidFittingPipe::landmarkSelection{
    5,
    6,
    7,
    8,
    9,
    10,
    11, // Chin
    // Others(frontal part)
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
};
} // namespace telef::align
