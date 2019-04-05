#pragma once

#include <vector>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace telef::intrinsic {
class IntrinsicDecomposition {
public:
	void initialize(const uint8_t *rgb, const uint8_t *normal, const uint16_t *depth, int width, int height);
	void process(float *result_intensity);
	void release();

private:
	void getMask(const uint16_t *depth);
	void getPoints(const uint16_t *depth);
	void getChrom();
	void getVarianceMap(int patch_size);
	void getGridLLEMatrix(int K, int g_size);
	void getNormalConstraintMatrix(float sig_n);
	void getContinuousConstraintMatrix(float sig_c, float sig_i);
	void pushSparseMatrix(CvSparseMat *src,Eigen::SparseMatrix<float>& tar);

	float *color;
	float *chrom;
	float *points;
	float *nMap;
	float *vMap;
	int *index;
	bool *mask;
	std::vector<std::pair<int,int> > indexMapping;

	Eigen::SparseMatrix<float> LLENORMAL;
	Eigen::SparseMatrix<float> LLEGRID;
	Eigen::SparseMatrix<float> WRC;
	Eigen::SparseMatrix<float> WSC;
	Eigen::SparseMatrix<float> MASK;
	Eigen::VectorXf consVecCont;

	int width;
	int height;
	int dims;
	};
}