#pragma once

#include <vector>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/png_io.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

namespace telef::intrinsic {
class IntrinsicDecomposition {
public:
	void initialize(const uint8_t *rgb, const uint8_t *normal, const uint16_t *depth, int width, int height);
	void process(double *result_intensity);
	void release();

private:
	void getMask(const uint16_t *depth);
	void getPoints(const uint16_t *depth);
	void getChrom();
	void getVarianceMap(int patch_size);
	void getGridLLEMatrix(int K, int g_size);
	void getNormalConstraintMatrix(double sig_n);
	void getContinuousConstraintMatrix(double sig_c, double sig_i);
	void getLaplacian();

	double *color;
	double *chrom;
	double *points;
	double *nMap;
	double *vMap;
	int *index;
	bool *mask;
	std::vector<std::pair<int,int> > indexMapping;

	Eigen::SparseMatrix<double> LLENORMAL;
	Eigen::SparseMatrix<double> LLEGRID;
	Eigen::SparseMatrix<double> WRC;
	Eigen::SparseMatrix<double> WSC;
	Eigen::SparseMatrix<double> MASK;
	Eigen::SparseMatrix<double> L_S;
	Eigen::VectorXd consVecCont;

	int width;
	int height;
	int dims;
	};
}