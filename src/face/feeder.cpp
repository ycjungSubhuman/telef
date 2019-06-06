#include "face/feeder.h"
#include "align/lmkfit_pipe.h"
#include "align/toresult_pipe.h"
#include "align/rigid_pipe.h"

#include <iostream>
#include <cmath>


namespace telef::face {

boost::shared_ptr<PCANonRigidAlignmentSuite>
MorphableModelFeederPipe::_processData(
    boost::shared_ptr<telef::feature::FittingSuite> in) {
  auto result = boost::make_shared<PCANonRigidAlignmentSuite>();
  result->fittingSuite = in;
  result->pca_model = pca_model;
  result->transformation = Eigen::Matrix4f::Identity();
  result->image = in->rawImage;
  result->fx = in->fx;
  result->fy = in->fy;
  result->shapeCoeff = Eigen::VectorXf::Zero(pca_model->getShapeRank());
  result->expressionCoeff =
    Eigen::VectorXf::Zero(pca_model->getExpressionRank());
  result->rawCloud = in->rawCloud;
  return result;
}

MorphableModelFeederPipe::MorphableModelFeederPipe(
    MorphableModelFeederPipe::MModelTptr model)
    : BaseT(), pca_model(model) {}

MultipleModelFeederPipe::MultipleModelFeederPipe(
  const std::vector<MModelTptr> &models, telef::io::MeshNormalDepthRendererWrapper render, float reg) : 
  BaseT(), pca_model_selected(nullptr), pca_models(models), reg(reg), render(render)
  {
  }

  boost::shared_ptr<PCANonRigidAlignmentSuite>
  MultipleModelFeederPipe::_processData(boost::shared_ptr<telef::feature::FittingSuite> in)
  {

    if(pca_model_selected == nullptr)
    {
      telef::align::LmkFitPipe fit(reg);
      telef::align::PCAToFittingResultPipe toresult;

      size_t minmodel = 0;
      float minerror = 1000000000.0f;

      for(size_t i=0; i<pca_models.size(); i++)
      {
        auto result = boost::make_shared<PCANonRigidAlignmentSuite>();
        result->fittingSuite = in;
        result->pca_model = pca_models[i];
        result->transformation = Eigen::Matrix4f::Identity();
        result->image = in->rawImage;
        result->fx = in->fx;
        result->fy = in->fy;
        result->shapeCoeff = Eigen::VectorXf::Zero(pca_models[i]->getShapeRank());
        result->expressionCoeff =
          Eigen::VectorXf::Zero(pca_models[i]->getExpressionRank());
        result->rawCloud = in->rawCloud;

        telef::align::PCARigidFittingPipe rigid;
        auto fitResult = fit(rigid(result));
        auto renderResult = render(toresult(fitResult));

        float error = 0.0f;
        float count = 0.0f;
        for(size_t j=0; j<renderResult->rendered_depth.size(); j++)
        {
          if(renderResult->rendered_depth[j] != 65535)
          {
            count += 1.0f;
            size_t row = j/in->rawImage->getWidth();
            size_t col = j%in->rawImage->getWidth();

            float raw_depth = in->rawCloud->at(col, row).z;
            if(std::isfinite(raw_depth))
            {
              float dist = raw_depth - static_cast<float>(renderResult->rendered_depth[j])/65535; 
              assert(count != 0.0f);
              error += dist*dist / count;
            }
          }
        }

        if(error < minerror)
        {
          minerror = error;
          minmodel = i;
        }
      }

      std::cout << std::endl << "Model " << minmodel << "Selected with error " <<  minerror << std::endl;
      this->pca_model_selected = pca_models[minmodel];
    }

    auto result = boost::make_shared<PCANonRigidAlignmentSuite>();
    result->fittingSuite = in;
    result->pca_model = this->pca_model_selected;
    result->transformation = Eigen::Matrix4f::Identity();
    result->image = in->rawImage;
    result->fx = in->fx;
    result->fy = in->fy;
    result->shapeCoeff = Eigen::VectorXf::Zero(pca_model_selected->getShapeRank());
    result->expressionCoeff =
      Eigen::VectorXf::Zero(pca_model_selected->getExpressionRank());
    result->rawCloud = in->rawCloud;
    return result;
  }

} // namespace telef::face