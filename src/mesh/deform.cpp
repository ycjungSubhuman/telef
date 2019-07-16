#include <Eigen/Sparse>
#include <vector>
#include "mesh/deform.h"
#include "igl/cotmatrix.h"
#include "util/require.h"

namespace telef::mesh
{

    namespace
    {
        Eigen::VectorXf SolveLS(
            const Eigen::SparseMatrix<double> &A,
            const Eigen::VectorXd &b)
        {
            Eigen::SparseMatrix<double> ATWA = A.transpose()*A;
            Eigen::VectorXd ATWb = A.transpose()*b;

            Eigen::SparseLU<
                Eigen::SparseMatrix<double>,
                Eigen::COLAMDOrdering<int>> lu;
            lu.analyzePattern(ATWA);
            lu.factorize(ATWA);
            TELEF_REQUIRE(Eigen::Success ==  lu.info());

            return lu.solve(ATWb).cast<float>();
        }
    }
    /* 
    Deform a mesh according to landmark constraint 

    linear least squares with soft laplacian constraint
    */
    Eigen::MatrixXf lmk2deformed(
        const Eigen::MatrixXf &V, 
        const Eigen::MatrixXi &F,
        telef::types::CloudConstPtrT landmark3d,
        const std::vector<int> &lmkinds,
        const float lmkweight)
    {
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V, F, L);
        Eigen::SparseMatrix<double> L_super(3*L.rows()+2*lmkinds.size()+1, 3*L.cols());

        // create diagonal L matrix (L_super) and add additinal rows for landmark constraints
        std::vector<Eigen::Triplet<double>> L_super_elems;
        //// Duplicate L three times
        for(int i=0; i<L.outerSize(); i++)
        {
            for(Eigen::SparseMatrix<double>::InnerIterator it(L,i);
                it; ++it)
            {
                int j = it.index();
                for (int k=0; k<3; k++)
                {
                    L_super_elems.emplace_back(L.rows()*k + i, L.cols()*k + j, it.value());
                }
            }
        }

        Eigen::VectorXd b_super(3*L.rows()+2*lmkinds.size()+1);

        //// Add landmark constraint
        for (size_t i=0; i<lmkinds.size(); i++)
        {
            for (int k=0; k<2; k++)
            {
                L_super_elems.emplace_back(3*L.rows()+2*i+k, L.rows()*k+lmkinds[i], lmkweight);
                if (k == 0)
                {
                    b_super(3*L.rows()+2*i+k) = lmkweight*landmark3d->points[i].x;
                }
                else
                {
                    b_super(3*L.rows()+2*i+k) = lmkweight*landmark3d->points[i].y;
                }
            }
        }
        L_super_elems.emplace_back(3*L.rows()+2*lmkinds.size(), 2*L.rows(), 1.0);
        b_super(3*L.rows()+2*lmkinds.size()) = V(0,2);

        // Create L_super
        L_super.setFromTriplets(L_super_elems.begin(), L_super_elems.end());
        Eigen::VectorXd V_flat = Eigen::Map<const Eigen::VectorXf>(V.data(), V.rows()*V.cols()).cast<double>();
        b_super.segment(0,3*L.rows()) = (L_super*V_flat).segment(0,3*L.rows());


        Eigen::VectorXf pos_flat = SolveLS(L_super, b_super);
        
        Eigen::MatrixXf pos = Eigen::Map<Eigen::MatrixXf>(pos_flat.data(), L.rows(), 3);

        return pos;
    }
}
