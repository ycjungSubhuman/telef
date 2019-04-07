#pragma once

#include <Eigen/Dense>
#include <tuple>

namespace telef::mesh
{
    // deformation weight of model
    template<typename T>
    using SingleWeight = Eigen::Array<T, 2, 1>;

    template<typename T>
    using Weights = std::vector<SingleWeight<T>>;

    template<typename T>
    using DefWeight = std::pair<Weights<T>, Weights<T>>;

    using GetWeightF = std::function<
        Weights<float>&(DefWeight<float> &w)>;

    namespace defw
    {
        template<typename T> inline auto
        create(const Weights<T> &wid, const Weights<T> &wex)
            -> DefWeight<T>
        {
            return std::make_pair(wid, wex);
        }

        template<typename T> inline auto
        id(const DefWeight<T> &dw)
            -> const Weights<T>& {
            return std::get<0>(dw);
        }

        template<typename T> inline auto
        exp(const DefWeight<T> &dw)
            -> const Weights<T>& {
            return std::get<1>(dw);
        }

        template<typename T> inline auto
        id(DefWeight<T> &dw)
            -> Weights<T>& {
            return std::get<0>(dw);
        }

        template<typename T> inline auto
        exp(DefWeight<T> &dw)
            -> Weights<T>& {
            return std::get<1>(dw);
        }

        template<typename T> inline auto
        id_only(const DefWeight<T> &dw)
            -> DefWeight<T> {
            const auto exSize = exp(dw).size();

            return create(id(dw), Weights<T>(exSize, SingleWeight<T>::Zero()));
        }

        template<typename T> inline auto
        exp_only(const DefWeight<T> &dw)
            -> DefWeight<T> {
            const auto idSize = id(dw).size();

            return create(Weights<T>(idSize, SingleWeight<T>::Zero()), exp(dw));
        }

        template<typename T, typename U> inline auto
        cast(const DefWeight<T> &dw)
            -> DefWeight<U>
        {
            const auto [a, b] = dw;

            DefWeight<U> result;
            for(size_t i=0; i<a.size(); i++)
            {
                id(result).emplace_back(a[i].template cast<U>());
            }
            for(size_t i=0; i<b.size(); i++)
            {
                exp(result).emplace_back(b[i].template cast<U>());
            }

            return result;
        }
    }
}
