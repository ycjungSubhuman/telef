#include "intrinsic/intrinsic.h"
#include <cstdio>

namespace telef::intrinsic{

void IntrinsicDecomposition::initialize(const uint8_t *_rgb, const uint8_t *_normal, const uint16_t *_depth, int _width, int _height)
{
	width = _width;
	height = _height;

	color = new float[3*width*height];
	chrom = new float[3*width*height];
	points = new float[3*width*height];
	nMap = new float[3*width*height];
	vMap = new float[width*height];
	mask = new bool[width*height];
	index = new int[width*height];

	getMask(_depth);
	getPoints(_depth);
	getChrom();
	int dims = indexMapping.size();
	LLENORMAL.resize(dims,dims);
	LLEGRID.resize(dims,dims);
	WRC.resize(dims,dims);
	WSC.resize(dims,dims);
	MASK.resize(dims,dims);
	consVecCont.resize(dims);
	for(int i=0;i<height;++i)
		for(int j=0;j<width;++j)
		{
			if(!mask[width*i+j])
				continue;

			float nn=0.0;
			for(int k=0;k<3;k++)
			{
				color[3*width*i+3*j+k]=_rgb[3*width*i+3*j+k]/255.0;
				nMap[3*width*i+3*j+k]=_normal[3*width*i+3*j+k]/128.0-1.0;
				nn+=nMap[3*width*i+3*j+k]*nMap[3*width*i+3*j+k];
			}
			nn = sqrt(nn);
			for(int k=0;k<3;k++)
				nMap[3*width*i+3*j+k]/=nn;
		}
	getVarianceMap(5);
}

void IntrinsicDecomposition::process(float *result_intensity)
{
	getGridLLEMatrix(10,6);
	getNormalConstraintMatrix(0.5);//sigma_n
	getContinuousConstraintMatrix(0.0001,0.8);//sigma_c,sigma_i

	//A = 4 * WRC + 3 * mask1 * (spI - LLEGRID) + 3 * mask2 * (spI - LLENORMAL) + 0.025 * WSC;
	//b = 4 * consVecCont;
	Eigen::SparseMatrix<float> spI(indexMapping.size(),indexMapping.size());
	spI.setIdentity();

	Eigen::SparseMatrix<float> A = 4 * WRC + 3 * MASK * (spI - LLEGRID) + 3 * MASK * (spI - LLENORMAL) + 0.025 * WSC;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > chol(A);  // performs a Cholesky factorization of A
	Eigen::VectorXf x = chol.solve(consVecCont*4);

	for(int it=0;it<indexMapping.size();it++)
	{
		int i=indexMapping[it].first;
		int j=indexMapping[it].second;
		result_intensity[i*width+j] = std::exp(x[it])/2.0;
	}
}

void IntrinsicDecomposition::release()
{
	delete [] color;
	delete [] chrom;
	delete [] points;
	delete [] nMap;
	delete [] vMap;
	delete [] mask;
	delete [] index;

	//cvReleaseSparseMat(&LLENORMAL);
	//cvReleaseSparseMat(&LLEGRID);
	//cvReleaseSparseMat(&WRC);
	//cvReleaseSparseMat(&WSC);
}

void IntrinsicDecomposition::getMask(const uint16_t *_depth)
{
	const uint16_t INVALID = 65535;

	int n=0;
	for (int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			mask[i*width+j] = (_depth[i*width+j]!=INVALID);
			if(mask[i*width+j])
			{
				index[i*width+j]=n++;
				indexMapping.push_back(std::make_pair(i,j));
			}
		}
}

void IntrinsicDecomposition::getPoints(const uint16_t *depth)
{
	float *is = new float[height];
	float *js = new float[width];

	for(int i=0;i<height;++i)
		is[i] = (float)(i-height/2)/height*2.0*std::tan(CV_PI/6);

	for(int j=0;j<width;++j)
		js[j] = (float)(j-width/2)/width*2.0*std::tan(CV_PI/6)*width/height;

	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			points[3*width*i+3*j+0]=js[j];
			points[3*width*i+3*j+1]=is[i];
			points[3*width*i+3*j+2]=-depth[i*width+j]/2000.0;
		}
	}
	delete [] is;
	delete [] js;
}

void IntrinsicDecomposition::getChrom()
{
	for(int it=0;it<indexMapping.size();++it)
	{
		int i=indexMapping[it].first;
		int j=indexMapping[it].second;

		float intensity = 0.0;
		intensity += color[3*width*i+3*j+0]*color[3*width*i+3*j+0];
		intensity += color[3*width*i+3*j+1]*color[3*width*i+3*j+1];
		intensity += color[3*width*i+3*j+2]*color[3*width*i+3*j+2];
		if(intensity<1e-10)
			intensity=1e-10;

		chrom[3*width*i+3*j+0] = color[3*width*i+3*j+0]/intensity;
		chrom[3*width*i+3*j+1] = color[3*width*i+3*j+1]/intensity;
		chrom[3*width*i+3*j+2] = color[3*width*i+3*j+2]/intensity;
	}
}

void IntrinsicDecomposition::getVarianceMap(int patch_size)
{
	int p_size=patch_size/2;
	for (int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			if(!mask[i*width+j])
				continue;
			int cnt=0;
			float p[3]={0.0,0.0,0.0},pp=0.0;
			for(int k=-p_size;k<=p_size;k++)
				for(int l=-p_size;l<=p_size;l++)
				{
					if(i+k<0 || i+k>=height || j+l<0 || j+l>=width || !mask[(i+k)*width+(j+l)])
						continue;
					for(int m=0;m<3;m++)
					{
						float var = nMap[3*(i+k)*width+3*(j+l)+m];
						p[m] += var;
						pp += var*var;
					}
					cnt++;
				}
			pp/=cnt;
			p[0]/=cnt;
			p[1]/=cnt;
			p[2]/=cnt;
			vMap[i*width+j] = pp-p[0]*p[0]-p[1]*p[1]-p[2]*p[2];
		}
	}
}

void IntrinsicDecomposition::getGridLLEMatrix(int K, int g_size)
{
	int ngrid_w = ceil(width / (float)g_size);
	int ngrid_h = ceil(height / (float)g_size);
	int Maxgrid = ngrid_w * ngrid_h;
	int *ipos = new int[Maxgrid];
	int *jpos = new int[Maxgrid];
	int Ngrid=0;

	for(int i=0;i<height;i+=g_size)
		for(int j=0;j<width;j+=g_size)
		{
			bool flag=false;
			for(int gi=0;gi<g_size && i+gi<height;gi++)
			{
				for(int gj=0;gj<g_size && j+gj<width;gj++)
					if(mask[(i+gi)*width+(j+gj)])
					{
						flag=true;
						break;
					}
				if(flag)
					break;
			}
			if(flag)
				Ngrid++;
		}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud3d (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud6d (new pcl::PointCloud<pcl::PointXYZRGB>);

  	// Generate pointcloud data
  	cloud3d->width = Ngrid;
  	cloud3d->height = 1;
  	cloud3d->points.resize (cloud3d->width * cloud3d->height);

  	cloud6d->width = Ngrid;	
  	cloud6d->height = 1;
  	cloud6d->points.resize (cloud6d->width * cloud6d->height);

	//building kd-tree
  	int n=0;
	for(int i=0;i<height;i+=g_size)
		for(int j=0;j<width;j+=g_size)
		{
			bool flag=true;
			double vmin = 99999999999;
			for(int gi=0;gi<g_size && i+gi<height;gi++)
				for(int gj=0;gj<g_size && j+gj<width;gj++)
				{
					if (i+gi<0 || i+gi>=height || j+gj<0 || j+gj>=width || !mask[(i+gi)*width+(j+gj)])
						continue;
					double var = vMap[(i+gi)*width+(j+gj)];
					if(mask[(i+gi)*width+(j+gj)] && var < vmin)
					{
						ipos[n] = i+gi;
						jpos[n] = j+gj;
						vmin = var;
						flag = false;
					}
				}
			if(flag)
				continue;
			//printf("%d / %d : %d %d -> %d -> %d\n",n,Ngrid,ipos[n],jpos[n],ipos[n]*width+jpos[n],index[ipos[n]*width+jpos[n]]);

			cloud3d->points[n].x = nMap[ipos[n]*width*3+jpos[n]*3+0];
			cloud3d->points[n].y = nMap[ipos[n]*width*3+jpos[n]*3+1];
			cloud3d->points[n].z = nMap[ipos[n]*width*3+jpos[n]*3+2];

			cloud6d->points[n].x = nMap[ipos[n]*width*3+jpos[n]*3+0];
			cloud6d->points[n].y = nMap[ipos[n]*width*3+jpos[n]*3+1];
			cloud6d->points[n].z = nMap[ipos[n]*width*3+jpos[n]*3+2];
			cloud6d->points[n].r = points[ipos[n]*width*3+jpos[n]*3+0];
			cloud6d->points[n].g = points[ipos[n]*width*3+jpos[n]*3+1];
			cloud6d->points[n].b = points[ipos[n]*width*3+jpos[n]*3+2];
			MASK.coeffRef(index[ipos[n]*width+jpos[n]],index[ipos[n]*width+jpos[n]]) = 1.0;
			n++;
		}


	std::vector<int> pointIdxNKNSearch(K+1);
	std::vector<float> pointNKNSquaredDistance(K+1);

	//for LLENORMAL
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree3d;
	kdtree3d.setInputCloud(cloud3d);
	cv::Mat1f z(K, 3);
	for(int i=0;i<Ngrid;i++)
	{
		float tol=1e-3;
		kdtree3d.nearestKSearch(cloud3d->points[i],K+1,pointIdxNKNSearch,pointNKNSquaredDistance);
		z.setTo(0);
		for(int k=1;k<pointIdxNKNSearch.size();++k)
		{
			z(k-1,0) = cloud3d->points[pointIdxNKNSearch[k]].x - cloud3d->points[i].x;
			z(k-1,1) = cloud3d->points[pointIdxNKNSearch[k]].y - cloud3d->points[i].y;
			z(k-1,2) = cloud3d->points[pointIdxNKNSearch[k]].z - cloud3d->points[i].z;
		}

		// % local covariance
		cv::Mat1f C = z * z.t(); // C = KxK matrix

		// % regularlization (K>D)
		float t = cv::trace(C)[0];
		C = C + tol*t*cv::Mat1f::eye(K, K);

		// % solve Cw=1
		cv::Mat1f w(K, 1);
		cv::solve(C, cv::Mat1f::ones(K, 1), w);
		float ws = 0;
		for(int k=0;k<K;k++)
			ws += w(k, 1);

		// % enforce sum(w)=1
		for(int k=0;k<K;k++) {
			int p = ipos[i]*width+jpos[i];
			int q = ipos[pointIdxNKNSearch[k]]*width+jpos[pointIdxNKNSearch[k]];
			//((float*)cvPtr2D(LLENORMAL, index[p], index[q]))[0] = w(k, 1) / ws;
			LLENORMAL.coeffRef(index[p],index[q]) = w(k,1) /ws;
		}
	}
	std::printf("LLENORMAL\n");

	//for LLEGRID
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree6d;
	kdtree6d.setInputCloud(cloud6d);
	for(int i=0;i<Ngrid;i++)
	{
		float tol=1e-3;
		kdtree6d.nearestKSearch(cloud6d->points[i],K+1,pointIdxNKNSearch,pointNKNSquaredDistance);
		z.setTo(0);
		for(int k=1;k<pointIdxNKNSearch.size();++k)
		{
			z(k-1,0) = cloud6d->points[pointIdxNKNSearch[k]].x - cloud6d->points[i].x;
			z(k-1,1) = cloud6d->points[pointIdxNKNSearch[k]].y - cloud6d->points[i].y;
			z(k-1,2) = cloud6d->points[pointIdxNKNSearch[k]].z - cloud6d->points[i].z;
		}

		// % local covariance
		cv::Mat1f C = z * z.t(); // C = KxK matrix

		// % regularlization (K>D)
		float t = cv::trace(C)[0];
		C = C + tol*t*cv::Mat1f::eye(K, K);

		// % solve Cw=1
		cv::Mat1f w(K, 1);
		cv::solve(C, cv::Mat1f::ones(K, 1), w);
		float ws = 0;
		for(int k=0;k<K;k++)
			ws += w(k, 1);

		// % enforce sum(w)=1
		for(int k=0;k<K;k++) {
			int p = ipos[i]*width+jpos[i];
			int q = ipos[pointIdxNKNSearch[k]]*width+jpos[pointIdxNKNSearch[k]];
			//((float*)cvPtr2D(LLEGRID, index[p], index[q]))[0] = w(k, 1) / ws;
			LLEGRID.coeffRef(index[p],index[q]) = w(k,1) /ws;
		}
	}
	std::printf("LLEGRID\n");
	delete [] ipos;
	delete [] jpos;
}

//WSC
void IntrinsicDecomposition::getNormalConstraintMatrix(float sig_n)
{
	int nx[] = {0, 0, 1, -1, -1, 1, 1, -1};
	int ny[] = {1, -1, 0, 0, -1, 1, -1, 1};
	float cp[3], cq[3];

	for(int it=0;it<indexMapping.size();++it)
	{
		int i=indexMapping[it].first;
		int j=indexMapping[it].second;

		if(!mask[i*width+j])
			continue;

		cp[0] = nMap[3*i*width+3*j+0];
		cp[1] = nMap[3*i*width+3*j+1];
		cp[2] = nMap[3*i*width+3*j+2];

		for(int k=0;k<8;k++)
		{
			int qi = i + nx[k];
			int qj = j + ny[k];
			if(qi < 0 || qj < 0 || qi >= height || qj >= width || !mask[qi*width+qj])
				continue;

			cq[0] = nMap[3*qi*width+3*qj+0];
			cq[1] = nMap[3*qi*width+3*qj+1];
			cq[2] = nMap[3*qi*width+3*qj+2];

			float dist = 2.0 * (1.0 - (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]));	

			float weight = (exp(-dist*dist/(sig_n*sig_n)));

			if(std::isnan(weight)) weight = 0;
			int p = index[i*width+j];
			int q = index[qi*width+qj];
			//((float*)cvPtr2D(WSC, p, p))[0] += weight;
			//((float*)cvPtr2D(WSC, q, q))[0] += weight;
			//((float*)cvPtr2D(WSC, p, q))[0] += -weight;
			//((float*)cvPtr2D(WSC, q, p))[0] += -weight;
			WSC.coeffRef(p,p) += weight;
			WSC.coeffRef(q,q) += weight;
			WSC.coeffRef(p,q) += -weight;
			WSC.coeffRef(q,p) += -weight;
		}
		//std::printf("%d / %d\n",it,indexMapping.size());
	}
	std::printf("WSC\n");
}

//WRC
void IntrinsicDecomposition::getContinuousConstraintMatrix(float sig_c, float sig_i)
{
	int nx[] = {0, 0, 1, -1, -1, 1, 1, -1};
	int ny[] = {1, -1, 0, 0, -1, 1, -1, 1};
	float cp[3], cq[3], ip[3], iq[3];
	float lp, lq;

	for(int it=0;it<indexMapping.size();++it)
	{
		int i=indexMapping[it].first;
		int j=indexMapping[it].second;

		if(!mask[i*width+j])
			continue;

		cp[0] = nMap[3*i*width+3*j+0];
		cp[1] = nMap[3*i*width+3*j+1];
		cp[2] = nMap[3*i*width+3*j+2];
		ip[0] = color[3*i*width+3*j+0];
		ip[1] = color[3*i*width+3*j+1];
		ip[2] = color[3*i*width+3*j+2];
		lp = std::sqrt(ip[0]*ip[0] + ip[1]*ip[1] + ip[2]*ip[2]);
		if(lp<0.00001)
			lp = 0.00001;
		lp = std::log(lp);

		for(int k=0;k<8;k++)
		{
			int qi = i + nx[k];
			int qj = j + ny[k];
			if(qi < 0 || qj < 0 || qi >= height || qj >= width || !mask[qi*width+qj])
				continue;

			cq[0] = nMap[3*qi*width+3*qj+0];
			cq[1] = nMap[3*qi*width+3*qj+1];
			cq[2] = nMap[3*qi*width+3*qj+2];
			iq[0] = color[3*qi*width+3*qj+0];
			iq[1] = color[3*qi*width+3*qj+1];
			iq[2] = color[3*qi*width+3*qj+2];
			lq = std::sqrt(iq[0]*iq[0] + iq[1]*iq[1] + iq[2]*iq[2]);
			if(lq<0.00001)
				lq = 0.00001;
			lq = std::log(lq);

			float dist = 2.0 * (1.0 - (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]));	
			float weight = (1 + exp(-exp(lp) * exp(lp) / (sig_i*sig_i) - exp(lq)*exp(lq) / (sig_i*sig_i)));

			weight = weight * (exp(-dist*dist/(sig_c*sig_c)));

			if(std::isnan(weight)) weight = 0;
			int p = index[i*width+j];
			int q = index[qi*width+qj];
			//((float*)cvPtr2D(WRC, p, p))[0] += weight;
			//((float*)cvPtr2D(WRC, q, q))[0] += weight;
			//((float*)cvPtr2D(WRC, p, q))[0] += -weight;
			//((float*)cvPtr2D(WRC, q, p))[0] += -weight;
			WRC.coeffRef(p,p) += weight;
			WRC.coeffRef(q,q) += weight;
			WRC.coeffRef(p,q) += -weight;
			WRC.coeffRef(q,p) += -weight;
		
			float dI = lp-lq;
			consVecCont.coeffRef(p) += weight * dI;
			consVecCont.coeffRef(q) -= weight * dI;
		}
		//std::printf("%d / %d\n",it,indexMapping.size());
	}
	std::printf("WRC, consVecCont\n");
}
}